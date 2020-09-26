

(defun random-from-boolean (p &aux (r (random 1.0d0))) (< r p))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Basic operations on integer distributions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defun factorial (n) (if (zerop n) 1 (* n (factorial (- n 1)))))

(defun poisson (k lambda)
  (/ (* (expt lambda k) (exp (- lambda))) (factorial k)))

(defun random-from-poisson (lambda &aux (s 0.0d0) (r (random 1.0d0)))
  (loop for k from 0 do
	(incf s (poisson k lambda))
	(when (<= r s) (return k))))

(defun random-element (l &aux (n (length l)) (r (random 1.0d0)))
  (if (null l) (error "random-element: argument is the empty list")
    (nth (floor (* r n)) l)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Tabulated functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defstruct table       ;a lookup-table of dimension 1 for a given 
                       ;function of 1 variable; eventually provide a 
                       ;default value/function for each dimension exceeded
  function              ;function stored in the table
  interval             ;step size for each dimension
  lower-bound          ;lower bound of each dimension
  upper-bound          ;upper bound of each dimension
  data                  ;array containing function values
  (initialized? nil)    ;initialization flag
  initializer          ;function to initialize the table
  )

(defvar *Phi*)

(defvar *Phi-inverse*)

(defun initialize-Phi-table (&aux (x1 0.0) y1 y2)
  "Computes integral of Normal-0-1 using trapezium rule, for now"
  (print "Initializing Phi table")
  (setf (aref (table-data *Phi*) 0) 0.5)
  (loop for i from 0 to (- (array-dimension (table-data *Phi*) 0) 2) do
    (setf y1 (normal-0-1 x1))
    (setf y2 (normal-0-1 (+ x1 (table-interval *Phi*))))
    (setf (aref (table-data *Phi*) (1+ i))
	  (+ (aref (table-data *Phi*) i) 
	     (* (table-interval *Phi*)
		(/ (+ y1 y2) 2.0))))
    (setf x1 (+ x1 (table-interval *Phi*))))
  (setf (table-initialized? *Phi*) t))

(defun initialize-Phi-inverse-table (&aux (just-bigger-x 0.0) just-bigger-y)
  "Computes inverse of Phi-integral"
  (print "Initializing Phi-inverse table")
  (setf (aref (table-data *Phi-inverse*) 0) -6.0)
  (setf (aref (table-data *Phi-inverse*) (round (/ 0.5 (table-interval *Phi-inverse*)))) 0)
  (setf (aref (table-data *Phi-inverse*) (round (/ 1.0 (table-interval *Phi-inverse*)))) 6.0)
  (loop for i from 0 to (- (round (/ 0.5 (table-interval *Phi-inverse*))) 2) do
    (let ((y (* (+ 501 i) (table-interval *Phi-inverse*))))
      (do ((x just-bigger-x (+ x (table-interval *Phi*))))
	  ((> (setf just-bigger-y (Phi-integral x)) y) (setf just-bigger-x x)))
      (setf (aref (table-data *Phi-inverse*) (+ 501 i))
	    (- just-bigger-x
	       (/ (* (- just-bigger-x (aref (table-data *Phi-inverse*) (+ 500 i))) (- y just-bigger-y))
		  (- (- y (table-interval *Phi-inverse*)) just-bigger-y))))))
  (loop for i from 0 to (- (round (/ 0.5 (table-interval *Phi-inverse*))) 2) do
    (setf (aref (table-data *Phi-inverse*) (- 499 i)) (- (aref (table-data *Phi-inverse*) (+ 501 i)))))
  (setf (table-initialized? *Phi-inverse*) t))

(unless (boundp '*Phi*)
  (setf *Phi* (make-table :interval 0.005 :lower-bound 0.0 :upper-bound 6.0
			  :data (make-array 
				 (list (1+ (round (/ 6.0 0.001))))
				 :initial-element 0)
			  :initializer #'initialize-Phi-table)))

(unless (boundp '*Phi-inverse*)
  (setf *Phi-inverse* 
    (make-table :interval 0.001 :lower-bound 0.0 :upper-bound 1.0
		:data (make-array 
		       (list (1+ (round (/ 1.0 0.001))))
		       :initial-element 0)
		:initializer #'initialize-Phi-inverse-table)))
  

(defun Phi-integral (x)
  "Computes the integral from -infinity to x of the normal curve N(0,1)"
  (cond ((< x 0) (- 1 (Phi-integral (- x))))
	((>= x (table-upper-bound *Phi*)) 1)
	(t (table-lookup x *Phi*))))


(defun Phi-inverse (X)  ;;; X is between 0 and 1
  (table-lookup X *Phi-inverse*))

(defun table-lookup (x       ;;; the actual input to the tabulated function
		      table   ;;; where the function is tabulated
		      &aux x1 ;;; the table point below x
		      dx ;;; x-x1 as a fraction of the interval
		      y1 ;;; the value of the integral at x1
		      y2 ;;; the value of the integral at x2
		      )
  (unless (table-initialized? table)
    (funcall (table-initializer table)))
  (multiple-value-setq (x1 dx) (floor (- x (table-lower-bound table)) (table-interval table)))
  (setf dx (/ dx (table-interval table)))
  (setf y1 (aref (table-data table) x1))
  (cond ((= dx 0) y1)
	(t (setf y2 (aref (table-data table) (1+ x1)))
	   (+ y1 (* dx (- y2 y1))))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Random sampling from continuous distributions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defun random-from-p (P-inverse &rest parameters)
  "To compute a random number from a parameterized probability distribution p,
   first define P(r) = \int_-\infty^r[p(x) dx]. Then if X is a uniformly-distributed
   random number in the range [0,1], r=P^-1(X) is a random number from the
   distribution p. We assume that P^-1 is available as the function P-inverse,
   which is defined to take the same parameters as the original p."
  (apply P-inverse (cons (random 1.0) parameters)))

(defun random-from-normal (mu sigma)
  (+ mu (* sigma (random-from-N-0-1))))

(defun random-from-N-0-1 ()
  (random-from-p #'Phi-inverse))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Basic operations on continuous distributions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defconstant *1-over-root-two-pi* (/ 1.0d0 (sqrt (* 2 pi))))

(defun normal-0-1 (x)
  ;Put in test for large x to avoid underflow error
  (if (> (abs x) 10)
      0.0d0
      (* *1-over-root-two-pi*
	 (exp (- (/ (* x x) 2))))))


(defun normal (x mu sigma &aux exponent)
  ;Put in test for large x to avoid underflow error
  (if (zerop sigma)
      (if (= x mu) most-positive-single-float 0d0)
    (if (> (setf exponent (/ (expt (- x mu) 2) (* 2 (expt sigma 2)))) 50)
	0d0
      (/ (* *1-over-root-two-pi* (exp (- exponent))) sigma))))

(defun logistic (x mu sigma)
  (/ 1 (+ 1 (exp (- (/ (- x mu) sigma))))))

(defun exponential-density (x beta)
  (if (< x 0) 0 (* beta (exp (- (* beta x))))))

(defun random-from-exponential (beta &aux (r (random 1.0d0)))
  (- (/ (log r) beta)))

;;; This is a very slow way to generate a sample.
;;; Faster would be to use binary search with the
;;; cdf for the innverse gamma, which involves the
;;; regularized gamma function, for which code is not written yet.

(defun random-from-density (f x0 dx &rest params)
  (let* ((s 0.0d0)
	 (r (random 1.0d0))
	 (ds 0)
	 (y0 0)
	 (y1 (apply f x0 params)))
    (loop for x from x0 by dx do
	  (setq y0 y1)
	  (setq y1 (apply f (+ x dx) params))
	  (setq ds (* dx (/ (+ y0 y1) 2.0d0)))
	  (when (> (+ s ds) r)
	    (return (+ x (* dx (/ (- r s) ds)))))
	  (setq s (+ s ds)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Gamma and Beta functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;; Lancoz Approximation to the Gamma Function                                                                                                                                                           

;;; from Press...                                                                                                                                                                                         

;;; Lancoz approximation for Re[z] > 0                                                                                                                                                                    
;;; Gamma(z+1) = (z+gamma+0.5)^{z + 0.5} exp{-(z+gamma+0.5)}                                                                                                                                              
;;;              * sqrt{2 \pi}[c_0 + {c_1 \over z+1} + ... + {c_N \over z+N} + eps]                                                                                                                       
;;;                                                                                                                                                                                                       
;;; for gamma=5, N=6, and a certain set of c_i, abs(eps)< 2.0E-10                                                                                                                                         
;;;  (even for complex arguments!)                                                                                                                                                                        

(defun gammln (xx)
  (if (< (realpart xx) 1.0)
      (let ((z (- 1.0 xx)))
        (- (log (/ (* pi z) (sin (* pi z))))
           (gammln (+ 1.0 z))))
      (let* ((z   (- xx 1.0))
             (tmp (+ z 5.0 0.5)))
        (+ (* (log tmp) (+ z 0.5))
           (- tmp)
           (log (sqrt (* 2 pi)))
           (log (+ 1.0
                   (/  76.18009173d0   (+ z 1.0d0))
                   (/ -86.50532033d0   (+ z 2.0d0))
                   (/  24.01409822d0   (+ z 3.0d0))
                   (/  -1.231739516d0  (+ z 4.0d0))
                   (/   0.120858003d-2 (+ z 5.0d0))
                   (/  -0.536382d-5    (+ z 6.0d0))))))))

(defun fgammln (xx)
  (declare (type single-float xx))
  (if (< (realpart xx) 1.0)
      (let ((z (- 1.0 xx)))
        (- (log (/ (* (float pi 1.0) z)
                   (sin (* (float pi 1.0) z)))) ; ***                                                                                                                                                     
           (fgammln (+ 1.0 z))))
      (let* ((z   (- xx 1.0))
             (tmp (+ z 5.0 0.5)))
        (+ (* (log tmp) (+ z 0.5))
           (- tmp)
           (float (log (sqrt (* 2 pi))) 1.0)
           (log (+ 1.0
                   (/  76.18009173e0   (+ z 1.0))
                   (/ -86.50532033e0   (+ z 2.0))
                   (/  24.01409822e0   (+ z 3.0))
                   (/  -1.231739516e0  (+ z 4.0))
                   (/   0.120858003e-2 (+ z 5.0))
                   (/  -0.536382e-5    (+ z 6.0))))))))


;;;; Gamma Function and Log Gamma Function                                                                                                                                                                

(defun gamma-function (a)
  (typecase a
    (integer      (exp (fgammln (float a))))
    (single-float (exp (fgammln        a )))
    (otherwise    (exp  (gammln        a )))))

(defun log-gamma-function (a)
  (typecase a
    (integer            (fgammln (float a)))
    (single-float       (fgammln        a ))
    (otherwise           (gammln        a ))))

(defun log-beta-function (m n)
  (- (+ (log-gamma-function m) (log-gamma-function n))
     (log-gamma-function (+ m n))))

(defun beta-function (m n)                      ;m,n>0                                                                                                                                                    
  (exp (log-beta-function m n)))

(defun inverse-gamma-density (x alpha beta)
  (if (<= x 0) 0.0d0
    (if (> (/ beta x) 103) 0.0d0
      (* (/ (expt beta alpha) (gamma-function alpha)) (expt x (- (+ alpha 1))) (exp (- (/ beta x)))))))

(defun random-from-inverse-gamma (alpha beta)
  (random-from-density #'inverse-gamma-density 0.0d0 0.001d0 alpha beta))
	
(defun gamma-density (x alpha beta)
  (* (/ (expt beta alpha) (gamma-function alpha)) (expt x (- alpha 1)) (exp (- (* beta x)))))

(defun random-from-gamma (alpha beta)
  (random-from-density #'gamma-density 0.0d0 0.001d0 alpha beta))



;;; Some of the  fixed parameters describe priors over the *physics*,
;;; which remains constant over all episodes. Hence the
;;; variables dependent on these should be sampled once to make a physics,
;;; and then multiple episodes should sampled from there.
;;; For example, Lambda_0 is the underlying rate of occurrence of events, which
;;; in reality is constant over episodes, although of course each episode gets 
;;; its own number of events sampled from the Poisson.
;;; From these episodes, a learning algorithm can narrow down on the
;;; actual value of variables such as Lambda_0.

(defvar J_0)
(defun J (x) (declare (ignore x)) J_0)
(defun I (x) (expt (J x) 2))
(defvar Lambda_0)
(defvar num-_events)
(defstruct _event id T_E L M)
(defvar _events)
(defvar num-stations)
(defstruct station id L_S N F nu sigma2 mu_t sigma2_t mu_a sigma2_a mu_n sigma2_n)
(defvar stations)
(defvar S1)
(defvar S2)
(defvar S3)
(defvar S4)
(defvar S5)
(defvar W_0)
(defun W (x) (declare (ignore x)) W_0)
(defun V (x) (expt (W x) 2))
(defun S (x y) (/ (abs (- x y)) (V 0.0)))
(defvar beta_0)
(defun beta (x) (declare (ignore x)) beta_0)
(defun alpha (x) (expt (beta x) 2))
(defun a (e s) (- (_event-M e) (* (alpha 0.0) (abs (- (station-L_S s) (_event-L e))))))
(defvar nu)
(defvar sigma2)
(defvar Detected)
(defvar Detections)
(defstruct detection station source T_D A_D s)


(defun generate-data (N file &key 
			(alpha_I 20.0)  (beta_I 2.0) ;;; gamma prior for event rate Lambda_0, mean is alpha/beta
			(alpha_No 3.0) (beta_No 2.0) ;;; inverse gamma prior for station noise levels, mean is beta/(alpha-1)
			(alpha_F 20.0) (beta_F 1.0)  ;;; gamma prior for station false alarm rate F, mean is alpha/beta
			(mu_V 5.0) (sigma2_V 1.0)    ;;; normal prior for W_0, sqrt of wave velocity V
			(mu_B 2.0) (sigma2_B 1.0)    ;;; normal prior for beta_0, sqrt of absorptivity alpha
			(mu_nu 1.0) (sigma2_nu 1.0)  ;;; normal prior for nu parameter of detection probability logistic function
			(alpha_s 2.0) (beta_s 1.0)   ;;; inverse-gamma prior for sigma2 parameter of detection probability logistic function
			(alpha_t 20.0) (beta_t 1.0)  ;;; normal-inverse-gamma prior for station arrival time uncertainty parameters mu_t and sigma2_t
			(delta_t 0.0) (lambda_t 1000.0)
			(alpha_a 2.0) (beta_a 1.0)   ;;; normal-inverse-gamma prior for station amplitude measurement uncertainty parameters mu_a and sigma2_a
			(delta_a 0.0) (lambda_a 1.0)
			(alpha_n 2.0) (beta_n 1.0)   ;;; normal-inverse-gamma prior for station noise amplitude distribution parameters mu_n and sigma2_n
			(delta_n 0.0) (lambda_n 1.0)
			)

  (setq Lambda_0 (random-from-gamma alpha_I beta_I))

  (setq num-stations 5)
  (setq S1 (make-station :id 'S1 :L_S 0.00)
	S2 (make-station :id 'S2 :L_S 0.25)
	S3 (make-station :id 'S3 :L_S 0.50)
	S4 (make-station :id 'S4 :L_S 0.75)
	S5 (make-station :id 'S5 :L_S 1.00))
  (setq stations (list S1 S2 S3 S4 S5))
  ;;; noise level N(s) drawn from an inverse gamma distribution InvGamma(αN,βN) 
  (loop for s in stations do (setf (station-N s) (random-from-inverse-gamma alpha_No beta_No)))
  ;;; false alarm rate F(s) drawn from a gamma distribution Gamma(αF,βF).
  (loop for s in stations do (setf (station-F s) (random-from-gamma alpha_F beta_F)))
  ;;; station detection parameters ν,σ  
  ;;; ν  drawn from a normal N(μ_v,σ_v), and 
  ;;; σ drawn from an inverse gamma InvGamma(αs,βs), 
  (loop for s in stations do (setf (station-nu s) (random-from-normal mu_nu (sqrt sigma2_nu))))
  (loop for s in stations do (setf (station-sigma2 s) (random-from-inverse-gamma alpha_s beta_s)))
  ;;; Uncertainty parameters for arrival time μ_t(s),σ_t(s) are fixed, unknown, station-specific
  ;;; parameters drawn from a normal-inverse-gamma distribution with parameters (delta_t, λt, αt, βt).
  (loop for s in stations do 
	(setf (station-sigma2_t s) (random-from-inverse-gamma alpha_t beta_t)))
  (loop for s in stations do 
	(setf (station-mu_t s) (random-from-normal delta_t (sqrt (/ (station-sigma2_t s) lambda_t)))))
  ;;; station amplitude uncertainty parameters μ_a(s),σ^2_a(s) 
  ;;; drawn from a normal-inverse-gamma distribution with parameters (delta_a, λa, αa, βa).
  (loop for s in stations do 
	(setf (station-sigma2_a s) (random-from-inverse-gamma alpha_a beta_a)))
  (loop for s in stations do 
	(setf (station-mu_a s) (random-from-normal delta_a (sqrt (/ (station-sigma2_a s) lambda_a)))))
  ;;; Noise amplitude prior parameters μ_n(s),σ^2_n(s) 
  ;;; drawn from a normal-inverse-gamma distribution with parameters (delta_n, λn, αn, βn).
  (loop for s in stations do 
	(setf (station-sigma2_n s) (random-from-inverse-gamma alpha_n beta_n)))
  (loop for s in stations do 
	(setf (station-mu_n s) (random-from-normal delta_n (sqrt (/ (station-sigma2_n s) lambda_n)))))

  ;;; nonuniform velocity V(x) in space, V = W^2, W ~ GP.
  ;;; For now, make V constant
  (setq W_0 (random-from-normal mu_V (sqrt sigma2_V)))

  ;;; nonuniform absorptivity alpha(x) in space, alpha = beta^2, beta ~ GP.
  ;;; For now, make alpha constant
  (setq beta_0 (random-from-normal mu_B sigma2_B))

  (loop for i from 1 to N do

	;;; create events or this episode
	(setq num-_events (random-from-poisson Lambda_0))
	(setq _events nil)
	(loop for e from 1 to num-_events do
	      (push (make-_event :id e) _events))
	(setq _events (reverse _events))

	;;; fix the times and locations of events
	(loop for e in _events do (setf (_event-L e) (random 1.0d0)))
	(loop for e in _events do (setf (_event-T_E e) (random 1.0d0)))
	;;; fix the magnitudes of events (starts at 2.0)
	(loop for e in _events do (setf (_event-M e) (+ 2 (random-from-exponential 2.302585))))


        ;;; Detection probability Logistic(A(e,s) - N(s),ν(s),σ(s)) 
        ;;; where Logistic(x,ν,σ) = 1/(1+exp(-(x-ν)/σ))
	(setq Detected (make-array (list num-_events num-stations)))
	(dotimes (n_e num-_events) 
	  (dotimes (n_s num-stations)
	    (let ((e (elt _events n_e)) (s (elt stations n_s)))
	      (setf (aref Detected n_e n_s) 
		    (random-from-boolean (logistic (- (a e s) (station-N s)) nu (sqrt sigma2)))))))

        ;;; create detections
	(setq detections nil)
	(dotimes (n_e num-_events) 
	  (dotimes (n_s num-stations)
	    (let ((e (elt _events n_e)) (s (elt stations n_s)))
	      (if (aref Detected n_e n_s) (push (make-detection :source e :station s) detections)))))

        ;;; number of noise detections d at station s is Poisson(N(s)):
	(loop for s in stations do
	      (let ((num-detections (random-from-poisson (station-N s))))
		(loop for n_d from 1 to num-detections do
		      (push (make-detection :source nil :station s) detections))))

	(setq detections (reverse detections))


        ;;; measured arrival time T_D(d) = T(e) + S(L(e), L(s)) + ε_t where		
        ;;; ε_t ∼ N(μ_t(s),σ_t(s)); uniform for noise detections
	(loop for d in detections do
	      (let ((e (detection-source d))
		    (s (detection-station d)))
		(setf (detection-T_D d)
		      (if (null e) (random 1.0d0)
			(random-from-normal (+ (_event-T_E e) (S (_event-L e) (station-L_S s)) (station-mu_t s))
					    (sqrt (station-sigma2_t s)))))))

        ;;; measured log amplitude A_D(d) = A(e,s) + ε_a where ε_a ∼ N(μ_a(s),σ^2_a(s))
        ;;; For noise detections, A_D(d) ∼ N(μ_n(s),σ^2_n(s))
	(loop for d in detections do
	      (let ((e (detection-source d))
		    (s (detection-station d)))
		(setf (detection-A_D d)
		      (if (null e) 
			  (random-from-normal (station-mu_n s) (sqrt (station-sigma2_n s)))
			(+ (a e s) (random-from-normal (station-mu_a s) (sqrt (station-sigma2_a s))))))))

        ;;; sign s(d) indicating the direction of arrival of the signal is +1 if L(e) < L(s), -1 otherwise.
        ;;; For noise detections s(d) is uniformly from {+1,-1}.
	(loop for d in detections do
	      (let ((e (detection-source d))
		    (s (detection-station d)))
		(setf (detection-s d)
		      (if (null e)
			  (random-element '(+1 -1))
			(if (< (_event-L e) (station-L_S s)) +1 -1)))))
	
	(print-episode i _events detections file)))

(defun print-episode (i events detections file)
    (with-open-file (stream file :direction :output :if-does-not-exist :create
                     :if-exists :append)
      (format stream "~%~%Episode ~A" i)
      (loop for e in events do
	    (print-event e stream))
      (loop for d in detections do
	    (print-detection d stream))))

(defun print-event (e stream)
  (format stream "~%Event :id ~3D :T_E ~10,6F :L_E ~10,6F :M ~10,6F" (_event-id e) (_event-T_E e) (_event-L e) (_event-M e)))

(defun print-detection (d stream)
  (format stream "~%Detection :station ~A :source ~3A :T_D ~10,6F :A_D ~10,6F :S ~A" 
	  (station-id (detection-station d))
	  (if (null (detection-source d)) nil (_event-id (detection-source d)))
	  (detection-T_D d)
	  (detection-A_D d)
	  (detection-s d)))