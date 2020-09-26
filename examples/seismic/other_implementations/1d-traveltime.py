# TODO:
#   using ctrl-C to fall into the debugger, test world state:  try adding extra events (e.g. the 16th event which disappears) and see how the world log probability changes.


# TODO: 
#   1. Give events unique id's we hash the colour from, which are assigned only at birth
#       Try giving them id's based on which potential event they came from
#   1b. Work out death probabilities; see if this fixed the isolated event or not

#   Different models:
#   -----------------
#   2. Consider the truncated normal model
#   3. Consider adding normal noise at each station; think GP.  equations still linear
#   
#   
#   Look at other todo's


# TODO:
# 
#       
#       1. Probabilistic model where the events are identifiable in principle:  one test -- the true state has highest prob, and cheat proposal will find it.  Add, for me, the magnitude range bars too.
#       2. Save the code into a new file, and refactor, rearrange.  Delete old crud.  Remove the hacks that weigh on the soul
#           Different optimisation now:  few iterations, so we can take the time to call world_d.  But it worries me that I need to!
#       3a.  Code up my accuracy measurement thing -- summarise the state's quality.
#       3. Think of better ways of finding potential events.  Summarise the trick I have already.  Start identifying /runs/
#       4. Better moves to explore the space, especially if (1) finds events which are /almost/ there but just require some fixing (death /won't/ kill the event, because it explains things well.)  Need to have compound moves to get anywhere, e.g. change assoc & repropose position.  Need to walk over areas of low probability.


# 
#   1. Add the magnitude noise bar too; it can be as close as the current one, colours will diambiguate.#   
#   Final state on noise 0.5 with cheat proposal is not entirely unreasonable; but it does /seem/ to get blips wrong, stuck in a local optima?
#       1 will help check this
# 
# 
#   Accuracy measure?  How to handle detecting more events than their should be?
#   See if the uniform proposal picks up lots of noise events?
#   Make world_d check to see that assoc is consistent.
#   Clean up the code!!!!!
#   Decrease the noise intensity log proportional (or something) to the number of noise events we have.
#       Or, have a noise 
#       
#   Try setting noise intensity to get average spacing noise spacing correct.


# TODO:
#   *       perhaps include supporting blips.  also consider runs and proposing events from them, and simple filtering by 
#   *       the maximum log-likelihood we can get by greedily grabbin blips
#   * events can get stuck into place (at least with incongruent standard deviations...) with the wrong bindings.
#           the correct event can only be allocated once the incorrect, but close, event is eliminated

# PUZZLES:
#   * Somehow there isn't a good proposal for the event at time 400...
#           Perhaps I need to start getting more robust potential events to solve this kind of problem.
#           There is also the problem of too many potential events, so they take a while to revist.
#   PRESCRIPTION:  prune potential events, and make them stronger
#       OTOH... don't want to miss them.  Maybe we need moves s.t. the potential events are just there to get things started.
#       or run for longer

# TODO:
#   1. Filter out by finding the best set of events locally, using loglikelihood
#   2. Investigate which events work
#   3. Generate sequences of images to disk (look up help(show)) of mcmc state.  Flick through them to see what's going on.
#   4. Reinstate cheat moves -- they suggest events which have probability closest to those of the
#       5. Or... we could bias the choice of potential events by their log likelihood

# CODE needed (don't forget to borrow)
#   inference accuracy measures (want to sample from posterior mode)

# MCMC moves:
#   first just start with a complete set, but one which doesn't allow much fluidity:
#     birth from a left/right singleton pair (or from prior), using the sampling method for selecting associations
#     death event
#     resample event parameters conditional on all blips (located down)
# 
#   birth event, from some number of left & right side blips sample the event using left_right_d
#   modify associations:  remove some of the left/right blips, then reselect a full set, then resample the event.
#   gibbs like move:  resample event parameters conditional on all its blips


# NOTES:
#   * To keep the posteriors normal, I've made magnitude decay linear


import numpy
import math
from time import time
from numpy import random
#import random
from math import exp, pi, sqrt

import scipy.special
import colorsys

random.seed(137)

DEBUG=False

# =========
# Utilities
# =========

rt2 = sqrt(2.)


def all(seq):
	for bb in seq:
		if not bb: return False
	return True

#def exponential(s):
#	return random.expovariate(1/s)

#def normal(m,s):
#	return random.normalvariate(m,s)

#random.exponential = exponential
#random.normal = normal

inf = float('inf')
nan = float('nan')

# take a list of tuples and return a tuple of lists
def unzip(seq):
    if len(seq)==0: return      # can't recover number of lists
    num = len(seq[0])
    lsts = [[] for i in range(num)]
    for t in seq:
        for i,x in enumerate(t):
            lsts[i].append(x)
    return tuple(lsts)

def maxargmax(seq, func=None):
    m = None
    am = None
    if func == None:
        for s, v in enumerate(seq):
            if m == None or m < v:
                am = s
                m = v
    else:
        # FIXME: this case is a little inconsistent:  func should be given the list index as a parameter
        for s in seq:                           
            v = func(s)
            if m == None or m < v:
                am = s
                m = v
    return m, am

def argmax(seq, **kwargs):
    return maxargmax(seq, **kwargs)[1]


def invert_dict(adict):
    inv = {}
    for v in adict.itervalues():
        inv[v] = [k for k in adict if adict[k] == v]
    return inv

# log which handles 0 ok (if I want it to)
def log(x, base=math.e, raise_on_zero=True):
    if x==0.0:
        if raise_on_zero: raise ValueError("Log of zero???")
        return -inf
    else:
        return math.log(x, base)

prod = lambda L: reduce(lambda x,y: x*y, L)

def logfact(x, lower=1):
    return sum([log(n) for n in xrange(lower,x+1)])

# compute log(sum([exp(x) for x in lst])) without arithmetic over/underflow;  from Nimar
def log_sum_exp(seq):
    m = max(seq)
    return m + log(sum([exp(x-m) for x in seq]))

def dist(x,y):
  return abs(x-y)

# Nimar's
class DefaultDict(dict):
  """
  Creates a dictionary with a default value function.
  Normally a None value returned by the default value function causes a
  key not found exception. However, if the none_valid argument is set to true
  then this feature is over-ridden, i.e. None values are returned by the
  dictionary.
  """
  def __init__(self, *args, **kwargs):
    """
    default - a function which takes the idx entry as the argument
    none_valid - defaults to False which indicates that if the default fn
                 returns None then a key error should be raised
    """
    # extract the default value from the keywrord arguments
    if "default" not in kwargs:
      raise "DefaultDict needs a default keyword argument"
    self.default = kwargs["default"]
    del kwargs["default"]
    
    # check for the none_valid argument
    if "none_valid" in kwargs:
      self.none_valid = True
      del kwargs["none_valid"]
    else:
      self.none_valid = False

    # pass the rest on to the dictionary base-type
    dict.__init__(self, *args, **kwargs)

  def copy(self):
    newdict = DefaultDict(default=self.default, none_valid=self.none_valid)
    for k,v in self.iteritems():
      newdict[k] = v.copy()
    return newdict
  
  def __getitem__(self, idx):
    try:
      return dict.__getitem__(self, idx)
    except KeyError:
      # if there is no default function then raise the exception
      if self.default is None:
        raise
      # try the default function
      val = self.default(idx)
      # re-raise the error if the default function returns None and
      # None is not a valid value
      if (val is None) and (not self.none_valid):
        raise
      # otherwise store the default in the index location and return
      # that value
      self[idx] = val
      return val
    
  def nodefault(self):
    """Disable the default function."""
    self.default = None


def list_product(lsts):
    if len(lsts)==0: 
        yield []
    else:
        for head in list_product(lsts[:-1]):
            for t in lsts[-1]:
                yield head + [t]





# ==============================
# Generic samplers and densities
# ==============================

def bernoulli_s(p):
    return (random.random() < p)

normal_s = random.normal
def normal_d(x, mean, std=None, var=None):
    if var != None:
        return -0.5 * log(2*pi * var) - (x-mean)**2/(2.0*var)
    elif std != None:
        return -0.5 * log(2*pi * std**2) - (x-mean)**2/(2.0*std**2)
    else:
        raise ValueError("normal_d: must pass in either variance or standard deviation")

poisson_s = random.poisson
def poisson_d(n, intensity):
    return -logfact(n) + n * log(intensity) - intensity

uniform_s = random.uniform
def uniform_d(x, l, u):
  if l <= x and x <= u:
    return - log(u-l)
  else:
    return -inf

def erfcc(x):
	"""Complementary error function."""
	z = abs(x)
	t = 1. / (1. + 0.5*z)
	r = t * exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
		t*(.09678418+t*(-.18628806+t*(.27886807+
		t*(-1.13520398+t*(1.48851587+t*(-.82215223+
		t*.17087277)))))))))
	if (x >= 0.):
		return r
	else:
		return 2. - r


def ncdf(x):
	"""Cumulative normal dist'n."""
	global rt2
	return 1. - 0.5*erfcc(x/rt2)


# this handles +inf and -inf without objection
def normal_cdf(x, mean, std):
#   return ncdf((x-mean)/std)
    return scipy.special.ndtr((x-mean)/std)

def normal_pdf(x, mean, std):
    # numpy.exp means we can pass arrays in
    return 1.0 / (sqrt(2*pi) * std) * numpy.exp(-(x-mean)**2/(2.0*std**2))


rejection_sample_calls = 0
rejection_sample_totalsamples = 0
# sample from a 1d density f(x) by finding a g(x) and M such that f(x) <= M g(x) and sampling from g(x)
#   input:  code to sample from g(x), and to compute f(x) / M g(x) i.e. the acceptance probability.
def rejection_sample(g_sampler, accept_prob):
    global rejection_sample_calls, rejection_sample_totalsamples
    rejection_sample_calls += 1
    
    for i in xrange(100000):
        x = g_sampler()
        p = accept_prob(x)

        if p > 1 or p < 0:
            raise ValueError("rejection_sample:  acceptance_prob out of range at %s with g_sampler=%s and accept_prob=%s" 
                    % (p, g_sampler, accept_prob))
        
        if (random.uniform() < p):
            rejection_sample_totalsamples += (i+1)
            return x
    
    raise ValueError("rejection_sample:  rejection sampler took too long using g_sampler=%s and accept_prob=%s" % (g_sampler, accept_prob))

def r1():
    print rejection_sample_totalsamples * 1.0/rejection_sample_calls 

def r2():
    global rejection_sample_calls, rejection_sample_totalsamples   
    rejection_sample_calls = 0
    rejection_sample_totalsamples = 0

def cond(i, c, e):
	if c:
		return i
	else:
		return e

# Truncated normal distribution 
def truncated_normal_s(mean, std, _lower=-inf, _upper=+inf):
    if _upper <= _lower:
        raise ValueError("truncated_normal_s: upper bound below lower (l=%s, u=%s)" % (lower, upper))
    
    lower = (_lower-mean)/std
    upper = (_upper-mean)/std
    
    # decide which rejection sampler to use; inspired by R msm code, but rederived most of the E[accep probability]'s too (semi-independently)
    #       I think I use less samples than msm?
    #           [it would be fun to also optimise the code of each sample i.e. cost of computing g and prob]
    #       actually, they do -- they almost double the size of the random exponential by not requiring it to touch at x=t.
    #           which has the effect of increasing the acceptance probability to  exp(- 0.5 * (x - lower)**2 ).
    #   use uniform over anything if (u-l) <= sqrt(2pi)
    #   always use exponential over normal, if possible
    #   exponential and uniform have their parameters optimised to maximise acceptance probability
    large_range = lower == -inf or upper == inf or (sqrt(2 * pi) < upper - lower)
    
    # FIXME: exponential graph doesn't quite align for tail probabilities.  It's not completely bogus, however.  We /can't/ do uniform, although it does work better on reasonable sized regions (e.g. 3).
    
    if 1 <= lower:
        if large_range:
            g = lambda: lower + random.exponential(lower/2.0)
            g.__name__ = "upper-tail-exponential"
            prob = lambda x: cond(exp(- 0.5 * x * (x - lower)), x <= upper, 0)
        else:
            g = lambda: uniform_s(lower, upper)
            g.__name__ = "upper-tail-uniform"
            prob = lambda x: exp(-x**2/2.0 + lower**2/2.0)
    elif upper <= -1:
        if large_range:
            g = lambda: upper - random.exponential(-upper/2.0)
            g.__name__ = "lower-tail-exponential"
            prob = lambda x: cond(exp(- 0.5 * x * (x - upper)), lower <= x, 0)
        else:
            g = lambda: uniform_s(lower, upper)
            g.__name__ = "lower-tail-uniform"
            prob = lambda x: exp(-x**2/2.0 + upper**2/2.0)        
    else:
        if large_range:
            g = lambda: normal_s()
            g.__name__ = "normal"
            prob = lambda x: cond(1, lower <= x and x <= upper, 0)
        else:
            g = lambda: uniform_s(lower, upper)
            g.__name__ = "uniform"
            if upper < 0:
                prob = lambda x: exp(-x**2/2.0 + upper**2/2.0)
                prob.__name__ = "case1"
            elif 0 < lower:
                prob = lambda x: exp(-x**2/2.0 + lower**2/2.0)
                prob.__name__ = "case2"
            else:                
                prob = lambda x: exp(-x**2/2.0)
                prob.__name__ = "case3"
    
    g.__name__ += " std normal with lower=%s, upper=%s" % (lower, upper)
    
    sample = mean + std * rejection_sample(g, prob)
    
    if sample < _lower or _upper < sample:
        raise ValueError("truncated_normal_s: my sample fell out of range at %s using g=%s and prob=%s! lower=%s upper=%s."
                % (sample, g, prob, lower, upper))
    
    return sample
    
# FIXME: fails quickly far out in the tails e.g. lower=30, upper=100, std=0.070710678118654752, mean=29.03248885547292
def truncated_normal_d(x, mean, std, lower, upper):
    if upper <= lower:
        raise ValueError("truncated_normal_s: upper bound below lower (l=%s, u=%s)" % (lower, upper))

    norm = normal_cdf(upper, mean, std) - normal_cdf(lower, mean, std)
    if norm > 0:
        return -log(sqrt(2*pi) * std) - (x-mean)**2/(2.0*std**2) - log(norm)
    else:
        return -inf



def normalise(ur_probs):
    total = 0
    for p in ur_probs:
        if p < 0:
            raise ValueError("normalise: given a negative probability!  ur_probs=%s" % (ur_probs,))
        total += p
    
    return [p/total for p in ur_probs]

def normalise_exp(ur_probs):
    m = max(ur_probs)
    total = 0
    for logp in ur_probs:
        p = exp(logp-m)
        total += p
    
    return [exp(logp-m)/total for logp in ur_probs]


# assumes probs sum to 1
def categorical_s(lst, probs, return_prob=False):
    if len(lst) != len(probs):
        raise ValueError("Error!  categorical_s:  given unaligned lists")
    
    sum = 0
    u = random.uniform()
    for i in range(len(lst)-1):
        sum += probs[i]
        if probs[i] < 0 or sum > 1 + 1e-5:
            raise ValueError("Error!  categorical_s:  either a probability below zero, or sums to more than 1 + epsilon")
        if u < sum:            
            if return_prob:
                return lst[i], probs[i]
            else:
                return lst[i]
    
    if return_prob:
        return lst[len(probs)-1], probs[-1]
    else:
        return lst[len(probs)-1]

def categorical_d(x, lst, probs, return_prob=False):
    if len(lst) != len(probs):
        raise ValueError("Error!  categorical_d:  given unaligned lists")
    
    if x not in lst:
        raise ValueError("Error!  categorical_d:  given element not in list")
    
    return log(probs[lst.index(x)])

# Arbitrary exponential scale
def exponential_s(shift, scale):
    return shift + random.exponential(scale)

def exponential_d(x, shift, scale):
    if x >= shift:
      return -log(scale) - (x-shift)/scale
    else:
      return -inf




# ===========================================
# Samplers and densities for earthquake model
# ===========================================


# Gutenberg-Richter sampling of magnitude
def gr_s(lower=0):
    return exponential_s(x, lower, 1/log(10))

def gr_d(x, lower=0):
    return exponential_d(x, lower, 1/log(10))


event_num_s = lambda: poisson_s(TOTAL_EVENT_INTENSITY)
event_time_s = lambda: uniform_s(TIME[0], TIME[1])
event_loc_s = lambda: uniform_s(SPACE[0], SPACE[1])
event_mag_s = lambda: EVENT_MAG_MIN + random.exponential(1/log(10))

def event_s():
    return (event_time_s(), event_loc_s(), event_mag_s())


event_num_d = lambda n: poisson_d(n, TOTAL_EVENT_INTENSITY)
event_time_d = lambda e: uniform_d(e[0], TIME[0], TIME[1])
event_loc_d = lambda e: uniform_d(e[1], SPACE[0], SPACE[1])
event_mag_d = lambda e: gr_d(e[2], EVENT_MAG_MIN)

def event_d(event):
    return event_time_d(event) + event_loc_d(event) + event_mag_d(event)


noise_num_s = lambda: poisson_s(TOTAL_NOISE_INTENSITY)
noise_num_d = lambda n: poisson_d(n, TOTAL_NOISE_INTENSITY)


def noise_s(detector_id):
    return (uniform_s(TIME[0], TIME[1] + TIME_EXTENSION), exponential_s(NOISE_MAG_MIN, NOISE_MAG_SCALE), detector_id)

# presume detector_id is set correctly
def noise_d(blip):
    t, m, this_id = blip
    return uniform_d(t, TIME[0], TIME[1] + TIME_EXTENSION) + exponential_d(m, NOISE_MAG_MIN, NOISE_MAG_SCALE)


def blip_s(event, detector_id):
    event_t, event_x, event_m = event
    d = dist(event_x, DETECTORS[detector_id]) 
    blip_t = event_t + d
    blip_m = event_m - MAG_DECAY * d
    return (random.normal(blip_t, TIME_STD_DEV), random.normal(blip_m, MAG_STD_DEV), detector_id)

# presume detector_id is set correctly
def blip_d(blip, event):
    t, m, detector_id = blip 
    event_t, event_x, event_m = event
    d = dist(event_x, DETECTORS[detector_id]) 
    t_mean = event_t + d
    m_mean = event_m - MAG_DECAY * d
    
    #print t, m, detector_id, event_t, event_x, event_m, d, t_mean
    
    return normal_d(t, t_mean, INF_TIME_STD_DEV) + normal_d(m, m_mean, INF_MAG_STD_DEV)



def event_blips_assoc_s(detector_id, events):
    blips = []
    assoc = {}
    for e in events:
        b = blip_s(e, detector_id)
        blips.append(b)
        assoc[b] = e    
    return blips, assoc

def noise_blips_s(detector_id):
    return [noise_s(detector_id) for i in range(noise_num_s())]

# Note:  noise events are greated last, so varying their number will not vary the events and true blips generated.  These can be varied by changing the random seed.
def world_s():
    true_events = [event_s() for i in range(event_num_s())]         
    true_events.sort()
    
    blips = []
    true_assoc = {}
    for detector_id in range(NUM_DETECTORS):
        event_blips, assoc = event_blips_assoc_s(detector_id, true_events)
        
        blips.append(event_blips)
        for k,v in assoc.iteritems():
            true_assoc[k] = v
   
# random.seed(SEED) # hack

    for detector_id in range(NUM_DETECTORS):
        blips[detector_id].extend(noise_blips_s(detector_id))
        blips[detector_id].sort()
        
    return (true_events, true_assoc, blips)


# computes log P(events, blips, assoc).  Checks world is consistent (i.e. assoc is)
def world_d(events, assoc, blips, return_components=False):
    logp_en = event_num_d(len(events))
    logp_e = sum([event_d(e) for e in events])
    
    logp_b = 0
    logp_bn = 0
    for blips_d in blips:
        num_noise = 0
        for b in blips_d:
            if b in assoc:
                logp_b += blip_d(b, assoc[b])
            else:
                logp_b += noise_d(b)
                num_noise += 1
        logp_bn += noise_num_d(num_noise)
    
    # check assoc is consistent:  each event associated to exactly one blip per detector, no blips repeated
    event_associated = [[False for n in range(NUM_DETECTORS)] for e in events]
    for b in assoc:
        b_detector = b[2]
        b_event = events.index(assoc[b])
        if event_associated[b_event][b_detector]:
            raise ValueError("world_d:  someone constructed an inconsistent world:  two blips associated to the same event.")
            return -inf
        else:
            event_associated[b_event][b_detector] = True
    if not all([all(event_assoc) for event_assoc in event_associated]):
        raise ValueError("world_d:  someone constructed an inconsistent world:  an event without its full complement of blips.")
        return -inf
        
    logp = logp_en + logp_e + logp_bn + logp_b
    
    #if DEBUG: 
    #print "world_d: logp_en=%s, logp_e=%s, logp_bn=%s, logp_b=%s, total=%s" % (logp_en, logp_e, logp_bn, logp_b, logp)
    
    if return_components:
        return logp, (logp_en, logp_e, logp_bn, logp_b)
    else:
        return logp



# ===================
# Posterior inference
# ===================



# Posterior over event loc/time/mag given a set of left blips and set of right blips
def left_right_d(left_blips, right_blips):
  if len(left_blips) == 0 or len(right_blips) == 0:
    raise ValueError("Error!  left_right_d:  need both left and right blips.  left_blips=%s, right_blips=%s." % (left_blips, right_blips))
  
  # find the boundary detectors, and check blips do not overlap.
  dleft = max([b[2] for b in left_blips])
  dright = min([b[2] for b in right_blips])
  left_x = DETECTORS[dleft]
  right_x = DETECTORS[dright]  
  
  if dright <= dleft:
    raise ValueError("Error!  left_right_d:  blips overlap.  left_blips=%s, right_blips=%s." % (left_blips, right_blips))
  
  # combine all left_blip observations to localisation of dleft's blip; similarly for right
  left_n = float(len(left_blips))
  left_t = sum([b[0] - (left_x - DETECTORS[b[2]]) for b in left_blips]) / left_n
  left_m = sum([b[1] + MAG_DECAY * (left_x - DETECTORS[b[2]]) for b in left_blips]) / left_n        # wrong wrong wrong  (??? FIXME!)
  right_n = float(len(right_blips))
  right_t = sum([b[0] - (DETECTORS[b[2]] - right_x) for b in right_blips]) / right_n
  right_m = sum([b[1] + MAG_DECAY * (DETECTORS[b[2]] - right_x) for b in right_blips]) / right_n

  # observations of time and magnitude  
  t = 0.5 * (left_t + right_t - (right_x - left_x))
  m = 0.5 * (left_m + right_m + MAG_DECAY * (right_x - left_x))
  
  std_dev_t = 1/2.0 * sqrt(INF_TIME_STD_DEV**2/left_n + INF_TIME_STD_DEV**2/right_n)
  std_dev_m = 1/2.0 * sqrt(INF_MAG_STD_DEV**2/left_n + INF_MAG_STD_DEV**2/right_n)
  
  # these are two independent observations of x, with standard deviations std_dev_t and std_dev_m respectively
  x1 = (left_t - right_t) / 2 + (right_x + left_x) / 2
  x2 = -(left_m - right_m) / (2 * MAG_DECAY) + (right_x + left_x) / 2

  std_dev_x1 = std_dev_t / sqrt(2)
  std_dev_x2 = std_dev_m / (sqrt(2) * MAG_DECAY)

  x = (x1 * std_dev_x1**-2 + x2 * std_dev_x2 **-2) / (std_dev_x1**-2 + std_dev_x2**-2)
  std_dev_x = (std_dev_x1**-2 + std_dev_x1**-2)**-0.5

  if DEBUG:
    out_of_range = (x < left_x or right_x < x or m < 0 or t < TIME[0] or TIME[1] < t)
    print "Potential event: from blips left=%s and right=%s, we have\n    mean (t=%.5f, x=%.5f, m=%.5f)\n    std-dev (%.5f, %.5f, %.5f)\n" \
        % ([print_blip(b) for b in left_blips], [print_blip(b) for b in right_blips], t, m, x, std_dev_t, std_dev_m, std_dev_x)
    if out_of_range:
      print "    ** mean out of range"
  
  # compute log likelihood of the mean event, log P(blips|event).  FIXME: really we want to compute an average over its uncertainty.
  # NOTE: this will still get bad'n's -- only detects local consistency
  #log_likelihood = sum([blip_d(b, (t, x, m)) for b in left_blips + right_blips])
  
  return ((t, x, m), (std_dev_t, std_dev_x, std_dev_m), (left_x, right_x))





# ===================
# Debugging utilities
# ===================


# Output various useful facts about the events -- whether the blips an event is associated with are very good, whether probability would go up if the event is deleted and why, etc.  Feel free to output a page of info.  Blips too.
def inspect_w(events, assoc, blips):   
    total_blips = sum([len(bps) for bps in blips])

    orig_prob, (logp_en, logp_e, logp_bn, logp_b) = world_d(events, assoc, blips, return_components=True)
    print "World (prob=%s) %s events, %s blips (%s real, %s noise)" \
        % (orig_prob, len(events), total_blips, NUM_DETECTORS * len(events), total_blips - NUM_DETECTORS * len(events))
    print "       logp_en=%s, logp_e=%s, logp_bn=%s, logp_b=%s" % (logp_en, logp_e, logp_bn, logp_b)
    
    
    # statistics about how blip log probability is decomposed
    num_n = []
    num_r = []
    logp_loc_n = []
    logp_loc_r = []
    logp_mag_n = []
    logp_mag_r = []    
    event_loc = [[0 for n in range(NUM_DETECTORS)] for e in events]
    event_mag = [[0 for n in range(NUM_DETECTORS)] for e in events]
    for blips_d in blips:
        num_noise = 0
        loc_n = 0.0
        loc_r = 0.0
        mag_n = 0.0
        mag_r = 0.0
        for blip in blips_d:
            if blip in assoc:
                t, m, detector_id = blip
                event_t, event_x, event_m = assoc[blip]
                d = dist(event_x, DETECTORS[detector_id]) 
                t_mean = event_t + d
                m_mean = event_m - MAG_DECAY * d
                
                b_event = events.index(assoc[blip])
                event_loc[b_event][detector_id] = normal_d(t, t_mean, INF_TIME_STD_DEV)
                event_mag[b_event][detector_id] = normal_d(m, m_mean, INF_MAG_STD_DEV)
                   
                loc_r += event_loc[b_event][detector_id]
                mag_r += event_mag[b_event][detector_id]
            else:
                t, m, this_id = blip
                loc_n += uniform_d(t, TIME[0], TIME[1] + TIME_EXTENSION)
                mag_n += exponential_d(m, NOISE_MAG_MIN, NOISE_MAG_SCALE)
                num_noise += 1
        num_n.append(num_noise)
        num_r.append(len(blips_d) - num_noise)
        logp_loc_n.append(loc_n)
        logp_mag_n.append(mag_n)
        logp_loc_r.append(loc_r)
        logp_mag_r.append(mag_r)
    
    if len(events)==0:
        num_r = [1] * NUM_DETECTORS
    
    print "Overall: loc_r = %.6f (%.6f); loc_n = %.6f (%.6f); mag_r = %.6f (%.6f); mag_n = %.6f (%.6f)" \
            % (sum(logp_loc_r), sum(logp_loc_r)/sum(num_r), sum(logp_loc_n), sum(logp_loc_n)/sum(num_n), \
               sum(logp_mag_r), sum(logp_mag_r)/sum(num_r), sum(logp_mag_n), sum(logp_mag_n)/sum(num_n))
    for d in range(NUM_DETECTORS):
        print "Detector %s: loc_r = %.6f (%.6f); loc_n = %.6f (%.6f); mag_r = %.6f (%.6f); mag_n = %.6f (%.6f)" \
            % (d, logp_loc_r[d], logp_loc_r[d]/num_r[d], logp_loc_n[d], logp_loc_n[d]/num_n[d], \
               logp_mag_r[d], logp_mag_r[d]/num_r[d], logp_mag_n[d], logp_mag_n[d]/num_n[d])
    
    
    invassoc = invert_dict(assoc)    
    for i, e in enumerate(events):
        our_blips = invassoc[e]
        our_blips.sort(cmp=lambda b1,b2: cmp(b1[2], b2[2]))
        
        llr = [(blip_d(b, e) - noise_d(b), b) for b in our_blips]
        optimal_llr = [maxargmax(detector_blips, lambda b: blip_d(b, e) - noise_d(b)) for detector_blips in blips]        
        
        newe = events[:]
        newa = assoc.copy()
        for k in assoc:
            if assoc[k] == newe[i]:
                del newa[k]
        del newe[i]
        delta = world_d(newe, newa, blips) - orig_prob
        
        print "Event %s: %s.  llr: %.6g=[%s].  prior_prob: %.6g." \
            % (i, print_event(e),  sum([x[0] for x in llr]), ", ".join(["%s:%.6g" % (blip_idx(b),v) for v,b in llr]), event_d(e))
        print "          delta world_p: %.6g  optimal_llr: %.6g=[%s]." \
            % (delta, sum([x[0] for x in optimal_llr]), ", ".join(["%s:%.6g" % (blip_idx(b),v) for v,b in optimal_llr]))
        print "          blip logp (loc, mag): %.6g = %.6g + %.6g = [%s]." \
            % (sum(event_loc[i])+sum(event_mag[i]), sum(event_loc[i]), sum(event_mag[i]), \
                ", ".join(["%.6g + %.6g" % (l,m) for l,m in zip(event_loc[i], event_mag[i])]))
    



"""
(i,e,a) = history[7]
inspect_w(e,a,blips)
sp = subplot(2,1,1); plot_world(sp,e,a,blips); sp = subplot(2,1,2); plot_world(sp,true_events,true_assoc,blips); show()              

check potential events

"""


# -2. Work out how pdb works.
# -1. Read the below... does it work?  On correct associations?
   #    Test this interactively:  (e,a,b) = world_s() ... grab an event and its blips... write useful code in the process
   
# 0. Check whether potential events finds points which assign high likelihood to the true events.
# 1. Debug the below -- does it work on correct associations?  
# 2. Check whether potential events is working correctly ....
# 3. Add filters to potential events -- if they suggest something which gives very low prob to the observations

# 4. Test inference on very low noise situation -- no noise blips, very low travel time etc noise.
#       What *should* happen, is we find exactly where the events are, and just propose them precisely.  Inference should work.

"""
(events, a, blips) = world_s()
ia = invert_dict(a)
e = events[0]
lb,rb = left_right_blips(e, ia)

left_right_d(lb, rb)

pdb.runcall(left_right_d, lb, rb)

"""



# test the truncated normal distribution sampler
def test_tn(lower, upper, num, plot_points=20):
    r2()
    tn = truncated_normal_s
    a = [tn(0, 1, lower, upper) for i in xrange(num)]
    
    size = 1.0 * (upper-lower)
    dx = size/plot_points
    
    x = [lower + i*1.0/plot_points * size for i in xrange(plot_points)]
    y1 = [0] * plot_points
    y2 = [0] * plot_points
    
    for v in a:
        if int(plot_points*(v-lower)/size) >= plot_points:
            print v, int(plot_points*(v-lower)/size), plot_points*(v-lower)/size
        y1[int(plot_points*(v-lower)/size)] += 1.0
    
    max_b = max(y1)
    min_b = min(y1)
    
    for i in xrange(plot_points):
        y1[i] /= num * dx
        y2[i] = exp(truncated_normal_d(x[i] + dx/2, 0, 1, lower, upper))
       
    sp = subplot(111)
    sp.set_title("Truncated standard normal (lower=%s, upper=%s) pdf vs. %s samples at %s points" % (lower, upper, num, plot_points)) 
    sp.plot(x,y1)
    sp.plot(x,y2)
    
    print "lower=%s, upper=%s.  total samples %s with %s buckets, each having between %s and %s samples."  \
        % (lower, upper, num, plot_points, max_b, min_b)
    r1()

def do_tests():
    # these cover all the cases
    figure(1)
    test_tn(-2,2,1000000)    
    figure(2)
    test_tn(-1,1,1000000)
    figure(3)
    test_tn(7,10,1000000)
    figure(4)
    test_tn(-10,-7,1000000)
    figure(5)
    test_tn(7,9,1000000)
    figure(6)
    test_tn(-9,-7,1000000)    
    figure(7)
    test_tn(-0.1,0.1,1000000)
    figure(8)
    test_tn(0.1,0.2,1000000)
    figure(9)
    test_tn(-0.2,-0.1,1000000)
    figure(10)
    test_tn(6.9,7,1000000)
    figure(11)
    test_tn(-7,-6.9,1000000)
    show()
    
    # still seems like too much noise in the uniform cases [-0.1, 0.1],  [0.1, 0.2],  and [-0.2, -0.1]
    # ... oh well, good enough for government work

  
"""

Event 2: (394.04835, 7.00412, 3.12165).  blip_llr: 27.0511070137=[7.52814, 7.55616, 2.80203, 3.12739, 6.0374].  prior_prob: -10.9590056677.  world_p delta: -15.6220977167

PEvent 57: (393.55434, 6.78993, 3.03605) blip_llr: 26.62=[7.81382, 7.53058, 2.02884, 3.38914, 5.85737] prior_prob: -10.76 std_dev=[1.41421, 0.209785, 0.212132] xran=[0, 30]
PEvent 60: (394.06870, 7.33792, 3.05321) blip_llr: 26.2=[7.307, 7.52134, 2.01215, 3.52968, 5.83466] prior_prob: -10.8 std_dev=[1.41421, 0.209785, 0.212132] xran=[0, 100]
PEvent 206: (397.18524, 12.16924, 2.51279) blip_llr: -4.646=[-15.1682, 6.17798, -3.60168, 4.3818, 3.56406] prior_prob: -inf std_dev=[1.41421, 0.209785, 0.212132] xran=[0, 40]

e1 = potential_events[57][0]

b1 = optimal_blips(e1, blips)
left_blips, right_blips = left_right_blips(e1, b1)

pot_desc = left_right_d(left_blips, right_blips)
event_mean, event_std, (x_left, x_right) = pot_desc


## Take this guy and find the blips it likes, the work out the posterior of that; does left_right_b work?

Yes, it appears so.  This event has a fair bit of noise, unfortunately, which does not get cancelled out.
...but, as a result, it seems way overconfident  about x localisation -- it's off by 5 with 0.16 stdev!


## Maybe I just need to use a sharper cut off for potential events as I've already computed them?

"""





# ==========================================================================================
# Compute potential event posterior means, conditioning on appropriately canopied blip pairs
# ==========================================================================================

def left_right_blips(e, our_blips):
    left = [b for b in our_blips if DETECTORS[b[2]] <= e[1]]
    right = [b for b in our_blips if e[1] < DETECTORS[b[2]]]          # TODO: asymmetry here is unsatisfying and unsettling 
    left.sort(cmp=lambda x,y: cmp(x[2], y[2]))
    right.sort(cmp=lambda x,y: cmp(x[2], y[2]))
    return left, right


def event_loglikelihood(event, blips):
    return sum([max([blip_d(b, event) - noise_d(b) for b in detector_blips]) for detector_blips in blips])

def optimal_blips(event, blips):
    arrivals = [[(blip_d(b, event) - noise_d(b), b) for b in detector_blips] for detector_blips in blips]
    our_arrivals = []
    for d_arrivals in arrivals:
        d_arrivals.sort(reverse=True)
        our_arrivals.append(d_arrivals[0][1])
    return our_arrivals



def compute_potential_events(blips, stats={}):
    station_pairs = [(s1,s2) for s1 in range(NUM_DETECTORS) for s2 in range(NUM_DETECTORS) if DETECTORS[s1] < DETECTORS[s2]]
    
    # TODO:  try only nearby stations
    station_pairs = [(s1,s2) for s1 in range(NUM_DETECTORS) for s2 in range(NUM_DETECTORS) if s1 < s2 and s2 <= s1 + 2]
    
    potential_events = []
    log_likelihoodratio_list = []

    # TODO:  CANOPY!
    #
    #window = {}
    #for i in range(num):
        #for j in range(num):
            #if i!=j: window[(i,j)] = dist(DETECTORS[i], DETECTORS[j])/VELOCITY
    
    #blip_sets = filter(
        #lambda l: all([ abs(l[i][0] - l[j][0]) <= window[(tuplet[i],tuplet[j])] for i in range(len(l)) for j in range(len(l)) if i!=j]), 
        #list_product([blips[i] for i in tuplet]))

    big_blips = [[b for b in blips[d] if b[1] >= POTENTIAL_EVENT_MAG_THRESHOLD] for d in range(NUM_DETECTORS)]
       
    for d in range(NUM_DETECTORS):        
        global true_assoc   # just for informational purposes, honest.
        num = len(blips[d])
        big_num = len(big_blips[d])
        num_true = len([b for b in blips[d] if b[2] == d and b in true_assoc])
        big_num_true = len([b for b in big_blips[d] if b[2] == d and b in true_assoc])
        print "Detector %s has %s=%s+%s blips filtered by magnitude to %s=%s+%s" \
            % (d, num, num_true, num-num_true, big_num, big_num_true, big_num-big_num_true)

    stats["compute_pot_events.total_blips"] = sum(len(bps) for bps in blips)
    stats["compute_pot_events.big_blips"] = sum(len(bps) for bps in big_blips)

    blip_pairs = 0
    in_range = 0
    are_likely = 0
    
    for d1, d2 in station_pairs:
        d_dist = DETECTORS[d2] - DETECTORS[d1] + TIME_STD_DEV*6
        #pot_pairs = list_product([blips[d] for d in [d1, d2]])
        
        pot_pairs = [(b1, b2) for (b1, b2) in list_product([big_blips[d] for d in [d1, d2]]) \
                        if abs(b1[0]-b2[0]) <= d_dist]
        
        print "Station pair (%s, %s) has %s possibilities from %s (could have been %s)." \
            % (d1, d2, len(pot_pairs), len(big_blips[d1]) * len(big_blips[d2]), len(blips[d1]) * len(blips[d2]))
        
        local_in_range = 0
        local_are_likely = 0
        local_accepted = 0
        local_time = time()
        for blip1, blip2 in pot_pairs:
            blip_pairs += 1
            
            pot_desc = left_right_d([blip1], [blip2])
            event_mean, event_std, (x_left, x_right) = pot_desc
                       
            if not (x_left - 3*event_std[1] <= event_mean[1] and event_mean[1] <= x_right + 3*event_std[1]):
                continue            
            if not (TIME[0] - 3*event_std[0] <= event_mean[0] and event_mean[0] <= TIME[1] + 3*event_std[0]):
                continue            
            if not (EVENT_MAG_MIN - 3*event_std[2] <= event_mean[2]):
                continue
            #if not (x_left <= event_mean[1] and event_mean[1] <= x_right):      # some problems with imagining the event outside of space
                #continue            
            #if not (TIME[0] <= event_mean[0] and event_mean[0] <= TIME[1]):
                #continue            
            #if not (EVENT_MAG_MIN  <= event_mean[2]):
                #continue
            
            local_in_range += 1
            
            
            #
            #   TODO: speed things up from here on down.
            #
            #
            
            arrivals = [maxargmax([blip_d(b, event_mean) - noise_d(b) for b in detector_blips]) for detector_blips in blips]
            
            log_likelihoodratio = sum([v for v, b in arrivals])            
            if log_likelihoodratio < POTENTIAL_EVENT_LLR_THRESHOLD1:
                continue
            
            local_are_likely += 1
            
            #potential_events.append((log_likelihoodratio, pot_desc))
            
            our_arrivals = [blips[d][x[1]] for d, x in enumerate(arrivals)]
            
            #left_blips, right_blips = left_right_blips(event_mean, our_arrivals)
            left_blips = [b for b in our_arrivals if b[2] <= blip1[2]]
            right_blips = [b for b in our_arrivals if blip2[2] <= b[2]]
            
            pot_desc = left_right_d(left_blips, right_blips)
            event_mean, event_std, (x_left, x_right) = pot_desc
                        
            if not (x_left - 3*event_std[1] <= event_mean[1] and event_mean[1] <= x_right + 3*event_std[1]):
                continue            
            if not (TIME[0] - 3*event_std[0] <= event_mean[0] and event_mean[0] <= TIME[1] + 3*event_std[0]):
                continue            
            if not (EVENT_MAG_MIN - 3*event_std[2] <= event_mean[2]):
                continue
            
            log_likelihoodratio = event_loglikelihood(event_mean, blips)        
            if log_likelihoodratio < POTENTIAL_EVENT_LLR_THRESHOLD2:
                continue             
            
            # NOTE:  various ways to find potential events; decode from the comments in the above code
            
            local_accepted += 1
            
            potential_events.append((log_likelihoodratio, pot_desc))
        
        local_time = time() - local_time
        print "     %s in range, %s are likely, %s accepted (time %.4gs)" % (local_in_range, local_are_likely, local_accepted, local_time)
        in_range += local_in_range
        are_likely += local_are_likely
        
        
    
    # FIXME:  a bunch of these proposed events will be duplicates:  one for each station pair.
    
    # pylab.hist([log(max(log_likelihoodratio_list)+1-x) for x in log_likelihoodratio_list], 100)

    stats["compute_pot_events.station_pairs"] = len(station_pairs)
    stats["compute_pot_events.blip_pairs"] = blip_pairs
    stats["compute_pot_events.in_range"] = in_range
    stats["compute_pot_events.are_likely"] = are_likely
    
    stats["compute_pot_events.accepted"] = len(potential_events)
    
    potential_events.sort(reverse=True)
    
    return [p for p in potential_events]
  
    # FIXME: too strong a filter that  x_left <= x <= x_right?





def compute_potential_events_old(blips, stats={}):
    station_pairs = [(s1,s2) for s1 in range(NUM_DETECTORS) for s2 in range(NUM_DETECTORS) if DETECTORS[s1] < DETECTORS[s2]]
    #station_pairs = [(s1,s1+1) for s1 in range(NUM_DETECTORS-1)]
    
    # FIXME:  proposing too many potential events... hard to get through them.  we want more reliable estimates, using longer runs than 1.
    
    potential_events = []
    log_likelihoodratio_list = []

    blip_pairs = 0
    in_range = 0
    for d1, d2 in station_pairs:
        for blip1, blip2 in list_product([blips[d] for d in [d1, d2]]):
            
            # TODO: canopy
            
            # FIXME:  if the observed time or magnitude different is too large, then the probabilitty of this observations supposing there is an event between these two stations becomes tiny; the x posterior mean may fall far out of the detector range (before truncation), unless the time and magnitude errors cancel out.
            
            blip_pairs += 1
            
            pot_desc = left_right_d([blip1], [blip2])
            event_mean, event_std, (x_left, x_right) = pot_desc
            
            # reasonable to be in range since it was proposed from this pair; some pair of blips will straddle the event
            # FIXME:  what is it's just a little bit outside?
            if not (x_left - 3*event_std[1] <= event_mean[1] and event_mean[1] <= x_right + 3*event_std[1]):
                continue
            
            if not (TIME[0] - 3*event_std[0] <= event_mean[0] and event_mean[0] <= TIME[1] + 3*event_std[0]):
                continue
            
            if not (EVENT_MAG_MIN - 3*event_std[2] <= event_mean[2]):
                continue
            
            in_range += 1
            
            log_likelihoodratio = event_loglikelihood(event_mean, blips)
            
            log_likelihoodratio_list.append(log_likelihoodratio)
           
            if log_likelihoodratio < POTENTIAL_EVENT_LLR_THRESHOLD:
                continue
            
            potential_events.append((log_likelihoodratio, pot_desc))
    
    # FIXME:  a bunch of these proposed events will be duplicates:  one for each station pair.
    
    # pylab.hist([log(max(log_likelihoodratio_list)+1-x) for x in log_likelihoodratio_list], 100)

    stats["compute_pot_events.station_pairs"] = len(station_pairs)
    stats["compute_pot_events.blip_pairs"] = blip_pairs
    stats["compute_pot_events.in_range"] = in_range
    stats["compute_pot_events.accepted"] = len(potential_events)
    
    potential_events.sort(reverse=True)
    
    return [p[1] for p in potential_events]
  
    # FIXME: too strong a filter that  x_left <= x <= x_right?




# ==============
# MCMC inference
# ==============

def mh_accept(logp, stats):
    stats["mh-moves"] += 1
    stats["mh-tot-logp"] += logp
    #print "mh_accept: logp=", logp   
    if logp >= 0:
        stats["mh-moves-accepted"] += 1
        return True
    else:
        accept = (random.rand() <= exp(logp))
        if accept: stats["mh-moves-accepted"] += 1
        return accept



MCMC_MOVES = [\
    "birthdeath_move which selects blips by sampling, using biased selection of potential events and death proposals",
    "for each pair of events in the same column,swap_event_pair_move",
    "for each event resample_move",
    "for each event reselect_blips_move",
]

mcmc_scans = []

def mcmc_scan(events, assoc, invassoc, blips, stats):  # or just invassoc?
    stats["scan"] += 1
    
    #biased_potevents = bernoulli_s(0.1)         # make inference worse
    birthdeath = birthdeath_move(events, assoc, invassoc, blips, stats, sample_blips=True, biased_potevents=True, biased_death=True)        
       
    swap_event_pair_accepted = 0
    
    close_events = [(events[e1i], events[e2i]) for e2i in range(len(events)) for e1i in range(e2i) \
                        if event_column(events[e1i], invassoc[events[e1i]]) == event_column(events[e1i], invassoc[events[e2i]]) and \
                           abs(events[e1i][0]-events[e2i][0]) <= abs(events[e1i][1]-events[e2i][1]) + 10 * TIME_STD_DEV ]    
    
    for e1, e2 in close_events:
        left_first = bernoulli_s(0.5)
        if swap_event_pair_move(e1, e2, left_first, events, assoc, invassoc, blips, stats):  swap_event_pair_accepted += 1
        if swap_event_pair_move(e1, e2, not left_first, events, assoc, invassoc, blips, stats):  swap_event_pair_accepted += 1
    
    resample_accepted = 0
    for idx in range(len(events)):
        if resample_move(idx, events, assoc, invassoc, blips, stats): resample_accepted += 1
        #adjust_move(idx, events, assoc, invassoc, blips, stats, bernoulli_s(0.5), bernoulli_s(0.5), bernoulli_s(0.5))
    
    reselect_accepted = 0
    for e in events:
        if reselect_blips_move(e, events, assoc, invassoc, blips, stats): reselect_accepted += 1
    
    global mcmc_scans
    moves_accepted = []
    if birthdeath: moves_accepted.append("birthdeath")
    if swap_event_pair_accepted > 0: moves_accepted.append("%s swap_event_pair" % swap_event_pair_accepted,)
    if resample_accepted > 0: moves_accepted.append("%s resample" % resample_accepted,)
    if reselect_accepted > 0: moves_accepted.append("%s reselect" % reselect_accepted,)
    mcmc_scans.append(", ".join(moves_accepted))
    
    #for e_id in random.permutation(len(events)):
      #for d_id in random.permutation(len(DETECTORS)):
        #swap_move(e_id, d_id, events, assoc, invassoc, blips, stats)





# ==========
# MCMC moves
# ==========


def event_quality(e, events, assoc, blips):
    newe = events[:]
    newa = assoc.copy()
    for k in assoc:
        if assoc[k] == e:
            del newa[k]
    del newe[events.index(e)]
    return world_d(newe, newa, blips) - world_d(events, assoc, blips)


# TODO: reinstate the different options
# TODO: intelligent death proposal too? -- prefer to kill the bad ones.  feels like some events will get stuck with this.
def birthdeath_move(events, assoc, invassoc, blips, stats, from_prior=False, sample_blips=True, biased_potevents=True, biased_death=True):
    do_death = len(events)>0 and bernoulli_s(DEATH_PROB)
   
    # 1. Construct the small world without the event in question
    # Two worlds:  (small_events, small_assoc) and (small_events + [event], small_assoc + arrivals).
    log_p_ratio = 0
    if do_death:
        stats["death_move"] += 1
        
        if biased_death:
            death_llr = [(event_quality(e, events, assoc, blips), e) for e in events]
            death_probs = normalise_exp([llr for llr, e in death_llr])
            event_idx = categorical_s(range(len(events)), death_probs)
        else:
            event_idx = random.randint(len(events))
        
        event = events[event_idx]
        
        small_events = events[:]
        small_assoc = assoc.copy()
        for k in assoc:
            if assoc[k] == event:
                del small_assoc[k]
        del small_events[event_idx]
    else:
        stats["birth_move"] += 1
        
        small_events = events
        small_assoc = assoc
        
    
    # 2. Compute the various things that the birth move needs to construct the large world -- this is used to compute log_q too.   
    unassoc =  [[b for b in stream if b not in small_assoc] for stream in blips]
    
    if biased_potevents:
        # score potential events by the log-likelihood of their mean
        # FIXME: should really integrate, taking into account the std-dev
        pot_ll = [(event_loglikelihood(pe[0], unassoc), pe) for pe in potential_events]
        #pot_ll.sort(reverse=True)
        pot_probs = normalise_exp([ll for ll, pe in pot_ll])
        
        our_pot_events = [pe for ll, pe in pot_ll]
    else:
        our_pot_events = potential_events
    
        
    # 3. Construct large world's extra event (if necessary)
    if do_death:
        t, x, m = event
    else:
        if biased_potevents:
            pot_event = categorical_s(pot_ll, pot_probs)[1]
        else:
            n = random.randint(len(potential_events))
            pot_event = potential_events[n]
        
        event_mean, event_std_dev, x_range = pot_event
        (t_mean, x_mean, m_mean) = event_mean
        (t_std_dev, x_std_dev, m_std_dev) = event_std_dev
        
        t = truncated_normal_s(t_mean, t_std_dev, TIME[0], TIME[1])
        x = truncated_normal_s(x_mean, x_std_dev, x_range[0], x_range[1])
        m = truncated_normal_s(m_mean, m_std_dev, EVENT_MAG_MIN, inf)
        event = (t, x, m)

            
    # 4. Compute what must be known about arrivals
    unassoc_llr = [[(blip_d(b, event) - noise_d(b), b)  for b in stream] for stream in unassoc]
    unassoc_prob = [] 
    for unassoc_d in unassoc_llr:
        #unassoc_d.sort(reverse=True)
        
        # unassoc is for the small world:  the first two should be impossible for a death move...?
        if len(unassoc_d) == 0:
            if do_death:
                raise ValueError("birthdeath_move:  death move with empty small unassoc_d.  This really shouldn't happen.")
            else:
                stats["birth-unassoc_empty"] += 1
                return False
        #elif max([exp(ratio) for ratio, b in unassoc_d]) == 0:      ## TODO: normalise_exp can handle this, however...
            #if do_death:
                #raise ValueError("birthdeath_move:  killing a really unlikely event; how'd that slip in?  I don't think this should happen.")
            #else:                
                #stats["birth-unassoc_really_unlikely"] += 1
                #return False
        else:
            probs = normalise_exp([ratio for ratio, b in unassoc_d])
            unassoc_prob.append(probs)
    
    
    # 5. Select (if necessary) large world's event arrivals
    if do_death:
        # Find the probability, in the small world, of the arrivals associated to event in the big world.
        arrivals = []
        arrivals_prob = []
        for unassoc_d, unassoc_prob_r in zip(unassoc_llr, unassoc_prob):
            for i in range(len(unassoc_d)):
                if assoc.get(unassoc_d[i][1], None) == event:
                    arrivals.append(unassoc_d[i][1])
                    arrivals_prob.append(log(unassoc_prob_r[i]))                ## FIXME: unassoc_prob_r[i] can be zero!
                    log_p_ratio += unassoc_d[i][0]  
                                
    else:
        arrivals = []
        arrivals_prob = []
        for unassoc_d, unassoc_prob_d in zip(unassoc_llr, unassoc_prob):
            (ratio, b), prob = categorical_s(unassoc_d, unassoc_prob_d, return_prob=True)
            arrivals.append(b)
            arrivals_prob.append(log(prob))
            log_p_ratio += ratio
        if len(arrivals) != NUM_DETECTORS:
            raise ValueError("birthdeath_event:  for birth, arrivals has wrong size.")
    
    # 5b. Some death things
    big_events = small_events + [event]
    big_assoc = small_assoc.copy()
    for b in arrivals:
        big_assoc[b] = event   
    
    if biased_death:
        death_llr = [(event_quality(e, big_events, big_assoc, blips), e) for e in big_events]
        death_probs = normalise_exp([llr for llr, e in death_llr])
        event_idx = len(big_events)-1
    
    
    # 6. Compute the /densities/ for the sample process, giving the log q(x|x')/q(x'|x) ratio.  Compute this for birth, negate for death
    log_q_death = log(DEATH_PROB)
    
    if biased_death:
        log_q_death += log(death_probs[event_idx])
    else:
        log_q_death += -log(len(small_events) + 1)
    
    log_q_birth = log(1-DEATH_PROB) + sum(arrivals_prob)
    
    # sum over all potential_events which could generate event, weighted by the probability of selecting that potential event
    log_q_birth_pot = []    
    for idx, ((t_mean, x_mean, m_mean), (t_std_dev, x_std_dev, m_std_dev), x_range) in enumerate(our_pot_events):
        if biased_potevents:
            logp = log(pot_probs[idx], raise_on_zero=False)
        else:
            logp = -log(len(potential_events))
            
        if -inf < logp:
            logp += truncated_normal_d(t, t_mean, t_std_dev, TIME[0], TIME[1]) + \
                    truncated_normal_d(x, x_mean, x_std_dev, x_range[0], x_range[1]) + \
                    truncated_normal_d(m, m_mean, m_std_dev, EVENT_MAG_MIN, inf)
        
        log_q_birth_pot.append(logp)
    
    log_q_birth += log_sum_exp(log_q_birth_pot)
    
    log_q = log_q_death - log_q_birth
    
    
    # 7. Compute the density ratio between big and small world  log p(x_big)/p(x_small) .  log_p_ratio is computed above.
    log_p = log(TOTAL_EVENT_INTENSITY) - log(len(small_events)+1) + event_d(event) + log_p_ratio
    
    # due to changing the number of noise blips
    for unassoc_d in unassoc:
        log_p += (-log(TOTAL_NOISE_INTENSITY) + log(len(unassoc_d)))

    # NOTE: log_p is now fixed
    
    
    # 8. Acceptance.
    log_a = log_q + log_p
    if do_death:
        log_a = -log_a
    
    if DEBUG_NEW:
        log_p2 = world_d(big_events, big_assoc, blips) - world_d(small_events, small_assoc, blips)
        print "birthdeath_move do_death=%s log_a=%.4g log_q=%.4g log_q_death=%.4g log_q_birth=%.4g log_p=%.4g log_p2=%.4g event_d=%.4g log_p_ratio=%4.g" \
                % (do_death, log_a, log_q, log_q_death, log_q_birth, log_p, log_p2, event_d(event), log_p_ratio)
        print "                arrivals_prob=%s" \
                % (arrivals_prob)            
    
    if mh_accept(log_a, stats):
        if do_death:
            #event_idx = events.index(event)
            #del events[event_idx]
            #for b in invassoc[event]:
                #if b in assoc: del assoc[b]         # FIXME: hack which shouldn't be required
            #del invassoc[event]
            stats["death_accepted"] += 1
        else:
            events.append(event)
            for b in arrivals:
                assoc[b] = event
            invassoc[event] = arrivals            
            stats["birth_accepted"] += 1
        return True
    return False
    
    # various options:
    #   order potential events by log-likelihood in /current world/.  sample from this normalised, or select the best
    #   sample from blips, or just select the best
    #   sample from potential events, or from prior (maybe).




# gibbs resample of event parameters; no change of associations
def resample_move(idx, events, assoc, invassoc, blips, stats):
    stats["resample"] += 1
    e = events[idx]    
    
    our_blips = invassoc[e]
    left_blips, right_blips = left_right_blips(e, our_blips)

    enew = event_posterior_s(left_blips, right_blips)
    
    log_q = event_posterior_d(e, left_blips, right_blips) - event_posterior_d(enew, left_blips, right_blips)
    
    log_p = event_d(enew) - event_d(e)
    log_p += sum([blip_d(b, enew) for b in our_blips])
    log_p -= sum([blip_d(b, e) for b in our_blips])
    
    log_a = log_q + log_p
    
    if DEBUG_NEW:
        print "resample_move:  log_q=%.4f log_p=%.4f log_a=%.4f" % (log_q, log_p, log_a)        
    
    ## always accept, because this is a Gibbs move -- actually, I don't think it is!
    ## The dependency structure is more complex due to identity uncertainty.  Let's compute the acceptance ratio proper like.
    if mh_accept(log_a, stats):        
        if DEBUG_NEW: w1 = world_d(events, assoc, blips)
        
        change_event_parameters(e, enew, events, assoc, invassoc)
        
        if DEBUG_NEW: w2 = world_d(events, assoc, blips)
        
        if DEBUG_NEW:
            print "   move accepted:  w_logp from %.4f to %.4f (%.4f)." % (w1, w2, w2-w1)
            print "   old_event=%s new_event=%s." % (print_event(e), print_event(enew))
        
        stats["resample_accepted"] += 1
        return True
    return False
# either uniformly select a close event, or uniformly select an arbitrary event
# def swap_move(e_id, d_id, events, assoc, invassoc, blips, stats):



    





# TODO: floor of Exp(lambda) is geometric distribution with probability 1-e^{-1/lambda}; but we'll do it the slow way for now
def truncated_geo_s(prob, maximum):
    for i in range(0, maximum):
        if bernoulli_s(prob):
            return i
    return maximum

def truncated_geo_d(x, prob, maximum):
    if x < 0 or x > maximum:
        return -inf
    elif x < maximum:
        return x * log(1-prob) + log(prob)
    else:
        return maximum * log(1-prob)


SWAP_SHIFT_PROB = 0.3

# Shifts the event left or right between detectors,
#     deletes all associations to the right (larger x coordinate),
#     then resamples them from left to right,

"""
TODO after:

def swap_right_move(e_idx, events, assoc, invassoc, blips, stats):
    event = events[e_idx]
    old_our_blips = invassoc[event]
    old_left_blips, old_right_blips = left_right_blips(event, our_blips)
           
    old_dleft = max([b[2] for b in old_left_blips)
    old_dright = min([b[2] for b in old_right_blips)
    
    # choose new event location; currently symmetric shift probabilities, truncated double geometric distribution
    if bernoulli_s(0.5):
        delta = truncated_geo_s(1-SWAP_SHIFT_PROB, NUM_DETECTORS - old_dright)
    else:
        delta = -truncated_geo_s(1-SWAP_SHIFT_PROB, old_dleft)
    
    dleft = old_dleft + delta
    dright = old_dright + delta
    
    # if we move the event to the left we remove left blips
    if delta < 0:
        left_blips = old_left_blips[0:dleft]
    else:
        left_blips = old_left_blips[:]
    right_blips = []
    
    # if we move the event right we need to sample new left blips.  we start with at least one left blip
    for detector in range(old_dleft+1, dleft+1):
        # just continue the left line

    # sample the right side blips; there must be at least one.  we start with all left blips, but perhaps no right blips
    for detector in range(dright, NUM_DETECTORS):
        
    
    # compute probabilities
    log_q = 0.0
    
    # print out proposal details
    if DEBUG_NEW:
        print "swap_left_move proposes:"
        print "Old blips: %s :: %s" % (print_blips(old_left_blips), print_blips(old_right_blips))
        print "Delta: %s ... " % (delta,)
        print "New blips: %s :: %s" %  (print_blips(left_blips), print_blips(right_blips))
        print "log_q: %s" % log_q,
"""


def event_column(event, blips):
    return len([b for b in blips if DETECTORS[b[2]] <= event[1]])


def event_posterior_s(left_blips, right_blips):
    (mean_t, mean_x, mean_m), (std_dev_t, std_dev_x, std_dev_m), (left_x, right_x) = left_right_d(left_blips, right_blips)
    
    t = truncated_normal_s(mean_t, std_dev_t, TIME[0], TIME[1])
    x = truncated_normal_s(mean_x, std_dev_x, left_x, right_x)
    m = truncated_normal_s(mean_m, std_dev_m, EVENT_MAG_MIN, inf)
    enew = (t, x, m)
        
    return enew

def event_posterior_d(event, left_blips, right_blips):
    (mean_t, mean_x, mean_m), (std_dev_t, std_dev_x, std_dev_m), (left_x, right_x) = left_right_d(left_blips, right_blips)
        
    (t, x, m) = event
    
    log_p = truncated_normal_d(t, mean_t, std_dev_t, TIME[0], TIME[1]) + \
            truncated_normal_d(x, mean_x, std_dev_x, left_x, right_x) + \
            truncated_normal_d(m, mean_m, std_dev_m, EVENT_MAG_MIN, inf)    
    
    return log_p




DEBUG_NEW = False

#  Swap either right or left side blips of a pair of events in the same column, then resample positions.
def swap_event_pair_move(e1, e2, swap_left, events, assoc, invassoc, blips, stats):
    stats["swap-event-pair-move"] += 1
    
    # find event between the same pair of detectors
    if event_column(e1, invassoc[e1]) != event_column(e1, invassoc[e2]):
        raise ValueError("swap_event_pair_move called with unswappable events")
    
    e1_left_blips, e1_right_blips = left_right_blips(e1, invassoc[e1])
    e2_left_blips, e2_right_blips = left_right_blips(e2, invassoc[e2])
    
    if swap_left:
        new_e1_left_blips = e2_left_blips
        new_e2_left_blips = e1_left_blips
        new_e1_right_blips = e1_right_blips
        new_e2_right_blips = e2_right_blips
    else:
        new_e1_left_blips = e1_left_blips
        new_e2_left_blips = e2_left_blips
        new_e1_right_blips = e2_right_blips
        new_e2_right_blips = e1_right_blips
    
    newe1 = event_posterior_s(e1_left_blips, e1_right_blips)
    newe2 = event_posterior_s(e1_left_blips, e1_right_blips)
    
    log_a = sum([blip_d(b, newe1) for b in new_e1_left_blips + new_e1_right_blips]) + \
            sum([blip_d(b, newe2) for b in new_e2_left_blips + new_e2_right_blips])
    log_a -= sum([blip_d(b, e1) for b in e1_left_blips + e1_right_blips]) + sum([blip_d(b, e2) for b in e2_left_blips + e2_right_blips])
    
    if DEBUG_NEW:
        pass
#        print "swap_event_pair %s e1=%s, e2=%s" % (cond("left", swap_left, "right"), print_event(e1), print_event(e2))
#        print "old_e1_blips=%s::%s, old_e2_blips=%s::%s" % \
#            (print_blips(e1_left_blips), print_blips(e1_right_blips), print_blips(e2_left_blips), print_blips(e2_right_blips))
#        print "new_e1_blips=%s::%s, new_e2_blips=%s::%s" % \
#            (print_blips(new_e1_left_blips), print_blips(new_e1_right_blips), print_blips(new_e2_left_blips), print_blips(new_e2_right_blips))
#        print "swap_event_pair_move log_a=%.4g" % log_a
    
    if mh_accept(log_a, stats):
        if DEBUG_NEW: w1 = world_d(events, assoc, blips)        
        
        change_event_parameters(e1, newe1, events, assoc, invassoc)
        change_event_parameters(e2, newe2, events, assoc, invassoc)
        change_event_blips(newe1, new_e1_left_blips + new_e1_right_blips, events, assoc, invassoc)
        change_event_blips(newe2, new_e2_left_blips + new_e2_right_blips, events, assoc, invassoc)
        
        if DEBUG_NEW: w2 = world_d(events, assoc, blips)
        
        if DEBUG_NEW:
            print "swap_event_pair_move log_q=%.4f log_p=%.4f log_a=%.4f" % (log_q, log_p, log_a)
            print "   move accepted:  w_logp from %.4f to %.4f (%.4f)." % (w1, w2, w2-w1)
        
        stats["swap-event-pair-move_accepted"] += 1
        return True    
    return False

def delete_event(event, events, assoc, invassoc):
    for b in invassoc[event]:
        del assoc[b]
    del invassoc[event]
    del events[events.index(event)]
        
def add_event(new_event, new_blips, events, assoc, invassoc):
    events.append(new_event)
    invassoc[new_event] = new_blips
    for b in new_blips:
        assoc[b] = new_event

def change_event_parameters(event, new_event, events, assoc, invassoc):
    events[events.index(event)] = new_event
    invassoc[new_event] = invassoc[event]
    del invassoc[event]
    for b in invassoc[new_event]:
        assoc[b] = new_event

def change_event_blips(event, new_blips, events, assoc, invassoc):
    for b in invassoc[event]:
        del assoc[b]
    for b in new_blips:
        assoc[b] = event
    invassoc[event] = new_blips




#check log_q for reselect -- the move made in the to get to final state looks bad, even though we have overall smaller logprob... (which may be due to something else)



# resample all blips conditional on event posterior given current blip assignment, then resample event parameters conditional on this.
# does swap move to resample things -- actually, doesn't yet, just uses unassociated blips.
def reselect_blips_move(event, events, assoc, invassoc, blips, stats):
    # actually, we'll relax the posterior in two ways:  forget truncation, and forget negative correlation between time and magnitude
    # TODO: write code to handle the negative correlation, i.e. general multivariate normals 
    stats["reselect-blips-move"] += 1
    
    old_blips = invassoc[event]
    old_left_blips, old_right_blips = left_right_blips(event, old_blips)
    
    log_q = 0.0
    
    # blips we, and the reverse direction, are allowed to associate with:  unassociated blips or those associated to event
    allowable_blips = [[b for b in blips[detector] if b not in assoc or assoc[b]==event] for detector in range(NUM_DETECTORS)]
    
    # select new blips
    new_blips = []
    (mean_t, mean_x, mean_m), (std_dev_t, std_dev_x, std_dev_m), (left_x, right_x) = left_right_d(old_left_blips, old_right_blips)    
    for detector in range(NUM_DETECTORS):
        d = dist(mean_x, DETECTORS[detector]) 
        
        t_mean = mean_t + d
        m_mean = mean_m - MAG_DECAY * d
        
        t_std_dev = INF_TIME_STD_DEV**2 + std_dev_t**2 + std_dev_x**2
        m_std_dev = INF_MAG_STD_DEV**2 + std_dev_m**2 + MAG_DECAY**2 * std_dev_x**2
               
        blip_probs = normalise_exp([normal_d(b[0], t_mean, var=t_std_dev) + normal_d(b[1], m_mean, var=m_std_dev) \
                                        for b in allowable_blips[detector]])
        new_blip, new_blip_prob = categorical_s(allowable_blips[detector], blip_probs, return_prob=True)
        new_blips.append(new_blip)
        
        log_q -= log(new_blip_prob)
    
    # if we're not changing blips do nothing since we're equivalent to an expensive resample_move
    if set(new_blips) == set(old_blips):
        stats["reselect-blips-move_same-blips"] += 1
        return False
    
    # resample event
    new_left_blips, new_right_blips = left_right_blips(event, new_blips)
    new_event = event_posterior_s(new_left_blips, new_right_blips)
    
    log_q -= event_posterior_d(new_event, new_left_blips, new_right_blips)
    
    # compute reverse probability
    (mean_t, mean_x, mean_m), (std_dev_t, std_dev_x, std_dev_m), (left_x, right_x) = left_right_d(new_left_blips, new_right_blips)    
    for blip in invassoc[event]:
        detector = blip[2]
        d = dist(mean_x, DETECTORS[detector])
        
        t_mean = mean_t + d
        m_mean = mean_m - MAG_DECAY * d
        
        t_std_dev = INF_TIME_STD_DEV**2 + std_dev_t**2 + std_dev_x**2
        m_std_dev = INF_MAG_STD_DEV**2 + std_dev_m**2 + MAG_DECAY**2 * std_dev_x**2
               
        blip_probs = normalise_exp([normal_d(b[0], t_mean, var=t_std_dev) + normal_d(b[1], m_mean, var=m_std_dev) \
                                        for b in allowable_blips[detector]])
        
        log_q += categorical_d(blip, allowable_blips[detector], blip_probs, return_prob=True)
    
    log_q += event_posterior_d(event, old_left_blips, old_right_blips)
    
    # compute world probability ratio -- where old_blips and new_blips overlap, the noise probabilities will cancel
    log_p = event_d(new_event) - event_d(event)
    log_p += sum([blip_d(b, new_event) for b in new_blips]) + sum([noise_d(b) for b in old_blips])
    log_p -= sum([blip_d(b, event) for b in old_blips]) + sum([noise_d(b) for b in new_blips])
                                        
    log_a = log_q + log_p
    
    if DEBUG_NEW:
        print "reselect_blips_move:  log_q=%.4f log_p=%.4f log_a=%.4f" % (log_q, log_p, log_a)
    
    if mh_accept(log_a, stats):
        
        if DEBUG_NEW: w1 = world_d(events, assoc, blips)
        
        change_event_parameters(event, new_event, events, assoc, invassoc)
        change_event_blips(new_event, new_blips, events, assoc, invassoc)
        
        if DEBUG_NEW: w2 = world_d(events, assoc, blips)
        
        if DEBUG_NEW:
            print "   move accepted:  w_logp from %.4f to %.4f (%.4f)." % (w1, w2, w2-w1)
            print "   old_event=%s old_blips=%s." % (print_event(event), print_blips(old_blips))
            print "   new_event=%s new_blips=%s." % (print_event(new_event), print_blips(old_blips))
        
        
        stats["reselect-blips-move_accepted"] += 1
        return True
    return False


#  Resamples all the right blips sequentially.  Runs are not explicitly identified, it is just supposed that there are a few choices of good blips compatible with the current left side run that repeating this enough will find the start of a run in short order (i.e. it will ignore all the small noise blips and go for the bigun's).

""""
def resample_right_move(e_idx, events, assoc, invassoc, blips, stats):
    event = events[e_idx]
    old_our_blips = invassoc[event]
    old_left_blips, old_right_blips = left_right_blips(event, our_blips)
           
    left_blips = old_left_blips
    right_blips = []
    
    # sample the first right blip; there must be one.
    
    
    # sample the rest of the right blips, if there are any.
    for detector in range(dright+1, NUM_DETECTORS):
        # the blip time/magnitude at the detector at position x_d are:
        #   t_d = t + (x_d - x)  + t_noise
        #       = t - x + x_d + t_noise
        #   m_d = m - MAG_DECAY * (x_d - x) + m_noise
        #       = m + MAG_DECAY * x - MAG_DECAY * x_d + m_noise
        # which is, unsurisingly, a correlated truncated normal distribution
        (mean_t, mean_x, mean_m), (std_dev_t, std_dev_x, std_dev_m), (left_x, right_x) = left_right_d(left_blips, right_blips)
        
        

    
    # compute new world, via swaps
    
    # compute probabilities
    log_q = 0.0
    
    # print out proposal details
    if DEBUG_NEW:
        print "swap_left_move proposes:"
        print "Old blips: %s :: %s" %  \
            (print_blips(old_left_blips)), \
             ", ".join(["%s:%s" % (blip_idx(b), print_blip(b)) for b in old_right_blips])
        print "New blips: %s :: %s" %  \
            (", ".join(["%s:%s" % (blip_idx(b), print_blip(b)) for b in left_blips]), \
             ", ".join(["%s:%s" % (blip_idx(b), print_blip(b)) for b in right_blips])
        print "log_q: %s" % log_q,


idea:  a bunch of little moves:
    special case resample for moves with only one detector to the right or left (so there are no runs to choose from)
        we can use general case code, without runs, for this too.  sampling proportional to the log give spread
    swap all left or right blips with another event, and resample both
    
    there are constraints, when we have magnitude /and/ time, to finding a second event.
"""


#  Resamples all the right blips sequentially.  Runs are not explicitly identified, it is just supposed that there are a few choices of good blips compatible with the current left side run that repeating this enough will find the start of a run in short order (i.e. it will ignore all the small noise blips and go for the bigun's).


# No time to finish in time

# Use data augmentation tricks to simplify
"""
def resample_right_move(e_idx, events, assoc, invassoc, blips, stats):
    event = events[e_idx]
    old_our_blips = invassoc[event]
    old_left_blips, old_right_blips = left_right_blips(event, our_blips)
           
    left_blips = old_left_blips
    right_blips = []
    
    log_q = 0
    
    # sample the first right blip; there must be one.
    # numerically integrate event_x between left_x and right_x.  start with a uniformly sampled offset.  use 1000 samples per blip.    
    d_left = left_blips[-1][2]
    d_right = right_blips[0][2]
    left_x = DETECTORS[d_left]
    right_x = DETECTORS[d_right]
    width = right_x - left_x
    dx = width/1000
    offset = uniform_s(0, dx) 
    
    left_n = float(len(left_blips))
    left_t_mean = sum([b[0] - (left_x - DETECTORS[b[2]]) for b in left_blips]) / left_n
    left_m_mean = sum([b[1] + MAG_DECAY * (left_x - DETECTORS[b[2]]) for b in left_blips]) / left_n        # wrong wrong wrong  (??? FIXME!)
    t_var = TIME_STD_DEV**2 / left_n
    m_var = MAG_STD_DEV**2 / left_n
    
    left_blip_t = left_blips[-1][0]
    left_blip_m = left_blips[-1][1]
    weights = []
    for b in blips[d_right]:
        blip_t = b[0]
        blip_m = b[1]
        x = offset
        v = 0.0
        for i in xrange(1000):
            time_difference = width - 2*x
            mag_difference = MAG_DECAY * (2*x - width)
            v += exp(-(blip_t - time_difference - left_blip_t)**2/(2*t_var) -(blip_m - mag_difference - left_blip_m)**2/(2*m_var))
            x += dx
        weights.append(v/1000)
    
    probs = normalise(weights)
    right_blips.append(categorical_s(blips[d_right], probs))
    
    log_q += log(probs[blips[d_right].index(old_right_blips[0])]) - log(probs[right_blips[0]]) 
    
    # sample the rest of the right blips, if there are any.
    for detector in range(dright+1, NUM_DETECTORS):
        (mean_t, mean_x, mean_m), (std_dev_t, std_dev_x, std_dev_m), (left_x, right_x) = left_right_d(left_blips, right_blips)
        
        # simplify by assuming the event is actually at its mean
        weights = [blip_d(b, (mean_t, mean_x, mean_m)) for b in blips[detector]]
        probs = normalise_exp(weights)
        
        new_blip, new_blip_prob = categorical_s(blips[detector], probs, return_prob=True)
        right_blips.append(new_blip)
        log_q -= log(new_prob)
    
    # compute reverse prob
    for i, detector in enumerate(range(dright+1, NUM_DETECTORS)):
        (mean_t, mean_x, mean_m), (std_dev_t, std_dev_x, std_dev_m), (left_x, right_x) = left_right_d(old_left_blips, old_right_blips[:i+1])
        
        # simplify by assuming the event is actually at its mean
        weights = [blip_d(b, (mean_t, mean_x, mean_m)) for b in blips[detector]]
        probs = normalise_exp(weights)
        log_q += log(probs[blips[detector].index(old_right_blips[i+1])])
    
    # construct new world, via swaps
    new_events = events[:]
    new_assoc = assoc.copy()
    
    # compute probabilities
    log_q = 0.0
    
    # print out proposal details
    if DEBUG_NEW:
        print "swap_left_move proposes:"
        print "Old blips: %s :: %s" %  \
            (print_blips(old_left_blips)), \
             ", ".join(["%s:%s" % (blip_idx(b), print_blip(b)) for b in old_right_blips])
        print "New blips: %s :: %s" %  \
            (", ".join(["%s:%s" % (blip_idx(b), print_blip(b)) for b in left_blips]), \
             ", ".join(["%s:%s" % (blip_idx(b), print_blip(b)) for b in right_blips])
        print "log_q: %s" % log_q,


idea:  a bunch of little moves:
    special case resample for moves with only one detector to the right or left (so there are no runs to choose from)
        we can use general case code, without runs, for this too.  sampling proportional to the log give spread
    swap all left or right blips with another event, and resample both
    
    there are constraints, when we have magnitude /and/ time, to finding a second event.


"""





# ============================
# Plotting and pretty printing
# ============================

def print_event(event):
    #loc = ", ".join(["%.5f" % (x,) for x in event[1]])
    return "(%.5f, %.5f, %.5f)" % (event[0], event[1], event[2])

def print_blip(blip):
    return "(%.5f, %.5f, %.5f)" % blip

def blip_idx(blip):
    return blips[blip[2]].index(blip)

def print_blips(bs):
    return "[%s]" % ", ".join(["%s:%s" % (blip_idx(b), print_blip(b)) for b in bs])

def print_invassoc(invassoc, events, blips):
    out = []
    for e in events:
        bs = invassoc[e]
        en = events.index(e)
        bsn = [(d,blips[d].index((t,m,d))) for t,m,d in bs]
        bsn.sort(lambda x,y: cmp(x[0], y[0]))        
        out.append("%s: [%s]" % (en, ", ".join(["%s" % bsni[1] for bsni in bsn])))
    return "{" + "; ".join(out) + "}"

def print_pevent(e):
    return "%s blip_llr: %.4g=%s prior_prob: %.4g std_dev=%s xran=%s" \
                % (print_event(e),  sum(llr), print_flist(llr), event_d(e), print_flist(pe[1]), print_flist(pe[2]))


def hash_as_set(seq):
    hashes = [hash(x) for x in seq]
    hashes.sort()
    out = hash(tuple(hashes))
    #print "Hashes=%s, out=%s" % (hashes, out)
    return out if out >= 0 else -out

# used to make consistent colours
def make_colour(seed):
    state = random.get_state()
    random.seed(seed)
    c = colorsys.hsv_to_rgb(uniform_s(), 1, 1)
    random.set_state(state)
    return c





def plot_world(sp, events, assoc, blips, show_events=True, mean_assoc_lines=None, event_blip_dt_size=0, event_blip_dm_size=0, scale=None, plot_noise_base=False, show_density=None, density_multipler=20, density_alpha=0.6):
    if scale == None:
        scale = FIG_SCALE
    if mean_assoc_lines == None:
        mean_assoc_lines = MEAN_ASSOC_LINES
    if show_density == None:
        show_density = SHOW_DENSITY
    
    #pylab.rc('lines', antialiased=False)
    #pylab.rc('collections', antialiased=False)
    colours = {}
    invassoc = invert_dict(assoc)
    for e in events:
        colours[e] = make_colour(hash_as_set(invassoc[e]))
        #if debug:
            #print "Event: %s has invassoc=%s with hash=%s colour=%s" \
                #% (print_event(e), tuple(invassoc[e]), hash(tuple(invassoc[e])), colours[e])
    
    # plot detectors
    for i in range(NUM_DETECTORS):
        sp.plot([TIME[0], TIME[1]+TIME_EXTENSION], [DETECTORS[i], DETECTORS[i]], c='k', linewidth=scale*1, zorder=-1) #, c='k', s=10)
        
    # plot association lines
    if show_events and len(events)>0:
        # either plot lines moving out of the event at the mean velocity, or 
        if mean_assoc_lines:
            for e in events:
                sp.plot([e[0], e[0] + (e[1]-SPACE[0])], [e[1], SPACE[0]], c=colours[e], linewidth=scale*2.5)
                sp.plot([e[0], e[0] + (SPACE[1]-e[1])], [e[1], SPACE[1]], c=colours[e], linewidth=scale*2.5)
        else:
            for b in assoc:
                e = assoc[b]
                sp.plot([e[0], b[0]], [e[1], DETECTORS[b[2]]], c=colours[e], linewidth=scale*2.5)
    
    # event-blip windows:  time and magnitude.
    if event_blip_dt_size>0:
        for e in events:
            for i in range(NUM_DETECTORS):
                dt = INF_TIME_STD_DEV * event_blip_dt_size
                mean_t = e[0] + dist(e[1], DETECTORS[i])
                mean_x = DETECTORS[i]
                sp.plot([mean_t - dt, mean_t + dt], [mean_x-1, mean_x-1], c=colours[e], linewidth=scale*0.5)
                sp.plot([mean_t - dt, mean_t - dt], [mean_x-1, mean_x-0.5], c=colours[e], linewidth=scale*0.5)
                sp.plot([mean_t + dt, mean_t + dt], [mean_x-1, mean_x-0.5], c=colours[e], linewidth=scale*0.5)
                sp.plot([mean_t, mean_t], [mean_x-1, mean_x], c=colours[e], linewidth=scale*0.5)                    
    if event_blip_dm_size>0:
        for e in events:
            for b in invassoc[e]:
                dm = INF_MAG_STD_DEV * event_blip_dm_size
                mean_m = e[2] - MAG_DECAY * dist(e[1], DETECTORS[b[2]])
                t_coord = b[0]
                x_coord = DETECTORS[b[2]] + 2 * mean_m
                sp.plot([t_coord + 0.5, t_coord + 0.5], [x_coord - 2*dm, x_coord + 2*dm], c=colours[e], linewidth=scale*0.5)
                sp.plot([t_coord, t_coord + 0.5], [x_coord - 2*dm, x_coord - 2*dm], c=colours[e], linewidth=scale*0.5)
                sp.plot([t_coord, t_coord + 0.5], [x_coord + 2*dm, x_coord + 2*dm], c=colours[e], linewidth=scale*0.5)
                sp.plot([t_coord + 0.25, t_coord + 0.5], [x_coord, x_coord], c=colours[e], linewidth=scale*0.5)                
    
                # 1. copy the blips code; get this part working
                # 2. start a new file.  combine all the comments, and refactor the code killing junk.
                # 3. move the parameters I want to modify into the correct place:  command line options (I alt-tab to the command line always)
    
    # plot blip base balls
    for bps in blips:
        if not plot_noise_base:
            bps = [b for b in bps if b in assoc]
        if len(bps) > 0:
            sizes = [scale**2 * 8**2 for b in bps]
            sp.scatter([b[0] for b in bps], [DETECTORS[b[2]] for b in bps], linewidth=[0 for b in bps], c='k', s=sizes, marker='o')
            b_colours = numpy.array([cond(colours[assoc[b]], (b in assoc), (0,0,0)) for b in bps])
            sizes = [scale**2 * 6**2 for b in bps]
            sp.scatter([b[0] for b in bps], [DETECTORS[b[2]] for b in bps], linewidth=[0 for b in bps], c=b_colours, s=sizes, marker='o')
        
    # plot events
    if show_events and len(events)>0:
        sp.scatter([e[0] for e in events], [e[1] for e in events], c=[colours[e] for e in events], s=[scale**2 * 6**2 * e[2]**2 for e in events], 
                linewidths=[0 for e in events], marker='s')
        sp.scatter([e[0] for e in events], [e[1] for e in events], c='k', s=scale**2 * 4**2, zorder = 3)
    
    # plot blips magnitudes
    for bps in blips:
        sorted_blips = bps[:]
        sorted_blips.sort(lambda x,y: cmp(x[1], y[1]), reverse=True)    # sorted by decreasing magnitude in case events overlap
        for b in sorted_blips :
            sp.plot([b[0], b[0]], [DETECTORS[b[2]], DETECTORS[b[2]] + 2 * b[1]], c='r', linewidth=scale*1.5)
            #sp.plot([b[0], b[0]], [DETECTORS[b[2]] + 2 * b[1], DETECTORS[b[2]] + 2 * b[1] + 1], c='m', linewidth=1)
    
    # print density
    if show_density:
        for e in events:
            for i in range(NUM_DETECTORS):
                std_dev = INF_TIME_STD_DEV
                dt = 0.1
                mean_t = e[0] + dist(e[1], DETECTORS[i])
                mean_x = DETECTORS[i]
                
                x = numpy.arange(mean_t - std_dev*3, mean_t + std_dev*3 + dt, dt)
                y = density_multipler * normal_pdf(x, mean_t, std_dev) + mean_x
                xv, yv = pylab.poly_between(x, mean_x, y)
                sp.fill(xv, yv, alpha=density_alpha, fc=colours[e], ec=colours[e])
    
    # enough room to see the detectors and the top most detector's arrival magnitudes
    extra_space = max(5, max([2*b[1] for b in blips[-1]]) + 1)
    
    # plot guide lines
    for x in [0]:
        sp.plot([x, x + SPACE_SIZE], [SPACE[0], SPACE[1]], ':', c='k', zorder=-1, linewidth=scale*2.5)
        sp.plot([x + SPACE_SIZE, x], [SPACE[0], SPACE[1]], ':', c='k', zorder=-1, linewidth=scale*2.5)
    
    sp.set_xlabel("Time")
    sp.set_ylabel("Space")
    sp.set_xlim([TIME[0], TIME[1]+TIME_EXTENSION])
    sp.set_xticks(range(TIME[0], TIME[1]+TIME_EXTENSION, SPACE_SIZE))
    sp.set_ylim([SPACE[0]-extra_space, SPACE[1]+extra_space])


"""
Old ideas:

for x in sp.axes.get_xticks():

# full grid of diagonal lines
for x in range(TIME[0], TIME[1], SPACE_SIZE):
   sp.plot([x, x + SPACE_SIZE], [SPACE[0], SPACE[1]], ':', c='k', zorder=-1)
   sp.plot([x + SPACE_SIZE, x], [SPACE[0], SPACE[1]], ':', c='k', zorder=-1)

for y in sp.axes.get_yticks():
   sp.plot([TIME[0], TIME[1]+TIME_EXTENSION], [y, y], ':', c='k', zorder=-1)

for x in [0]:
    sp.plot([x, x + SPACE_SIZE+10], [SPACE[0]-5, SPACE[1]+5], ':', c='k', zorder=-1)
    sp.plot([x + SPACE_SIZE+10, x], [SPACE[0]-5, SPACE[1]+5], ':', c='k', zorder=-1)
"""



def plot_w():
    global events, assoc, blips
    figure()
    sp = pylab.subplot(2,1,1)
    sp.set_title("True world state")
    plot_world(sp, true_events, true_assoc, blips)
    sp = pylab.subplot(2,1,2)
    sp.set_title("Potential event locations")
    plot_world(sp, events, assoc, blips)

    



# =======
# Globals
# =======

potential_events = []
blips = []
events = []
assoc = {}
invassoc = {}
stats = DefaultDict(default=lambda x: 0.0)
    





# ==================
# Error measurements
# ==================

# define an event to be correctly located if it is within so-and-so
#       what if there are multiple guesses corresponding to the same event?  currently we won't pick up on this.
#           record an average multiplicity?
#       set a threshold for closeness, in terms of # of std-deviations

# precision:  fraction of the guessed events are correctly located true events
# recall:  fraction of true events are correctly located
# f-measure

# NOTE:  could just care about event position not magnitude... in fact, we do!  This will just measure positional localisation.
def corresponds_to(guessed_event, true_event, true_blips, slack=2):
    left_blips, right_blips = left_right_blips(true_event, true_blips)
    e_mean, (t_std_dev, x_std_dev, m_std_dev), e_xrange = left_right_d(left_blips, right_blips)
    
    # problem: this becomes harder at lower standard deviations...
    # Hack: a little bit bigger than potential events with time_std_dev=2 mag_std_dev=0.3
    #t_std_dev = 1
    #x_std_dev = 0.5
    
    # better:
    #t_std_dev = max(1, t_std_dev)
    #x_std_dev = max(0.5, s_std_dev)

    return abs(guessed_event[0] - true_event[0]) <= slack * t_std_dev and \
           dist(guessed_event[1], true_event[1]) <= slack * x_std_dev


def precision(guessed_events, true_events, true_invassoc):
    guesses = 0.0
    correct = 0.0
    for e_guess in guessed_events:
        guesses += 1        
        for e_true in true_events:
            if corresponds_to(e_guess, e_true, true_invassoc[e_true]):
                correct += 1
                continue
    if guesses > 0:
        return correct/guesses
    else:
        return 1.0

def recall(guessed_events, true_events, true_invassoc):
    true = 0.0
    guessed = 0.0
    for e_true in true_events:    
        true += 1        
        for e_guess in guessed_events:    
            if corresponds_to(e_guess, e_true, true_invassoc[e_true]):
                guessed += 1
                continue
    if true > 0:
    	return guessed/true
    else:
    	return 1.0

def f1(guessed_events, true_events, true_invassoc):
    prec = precision(guessed_events, true_events, true_invassoc)
    rec = recall(guessed_events, true_events, true_invassoc)    
    
    if (prec+rec) != 0:
    	return 2.0 * prec * rec / (prec + rec)
    else:
    	return 0.0

def precisionb(assoc, true_assoc):
    guesses = 0.0
    correct = 0.0
    
    # Precision of the relation {(b1,b2) : b1 and b2 come from the same event}
    # Loop over all pairs of blips predicted to be non-noise and from the same event
    # Prediction is correct if these are both really non-noise, and we are correct about whether they come from the same event or not.
    for b1 in assoc:
        for b2 in assoc:
            if assoc[b1] == assoc[b2]:
                guesses += 1
                if b1 in true_assoc and b2 in true_assoc and true_assoc[b1] == true_assoc[b2]:
                    correct += 1
    
    if guesses > 0:
        return correct/guesses
    else:
        return 1.0

def recallb(assoc, true_assoc):
    return precisionb(true_assoc, assoc)

def f1b(assoc, true_assoc):
    prec = precisionb(assoc, true_assoc)
    rec = recallb(assoc, true_assoc)
    
    if (prec+rec) != 0:
    	return 2.0 * prec * rec / (prec + rec)
    else:
    	return 0.0


# ====
# Main
# ====
def print_flist(lst):
    return "[" + ", ".join(["%.6g" % (x,) for x in lst]) + "]"

def figure():
    fig = pylab.figure()
    fig.subplots_adjust(left=0.045, bottom=0.055, right=0.99, top=0.98)
    return fig

def safe_div(a, b):
    if b==0:
        return nan
    else:
        return a/b

def main():
    # set hyperparameters place here.
    global blips, true_events, true_assoc
    global events, assoc, invassoc, stats
    global potential_events, history, full_history, mcmc_scans
    
    (true_events, true_assoc, blips) = world_s()
    
    print "True events (%s):"  % (len(true_events),)
    inspect_w(true_events, true_assoc, blips)
    
    #print "True events (%s): %s" % (len(true_events), ", ".join([print_event(e) for e in true_events]))
    print
    print "Blips"
    for i, bps in enumerate(blips):
        #print "# Detector %s @ %s has %s events: %s" % (i, DETECTORS[i], len(bps), ", ".join([print_blip(b) for b in bps]))
        print "# Detector %s @ %s has %s events." % (i, DETECTORS[i], len(bps))
        print
    print
    
    ## True state 
    #if not GRAPHS_LATER:
        #figure()
        #sp = pylab.subplot(1,1,1)
        ##sp.set_title("True world state")        
        #plot_world(sp, true_events, true_assoc, blips)   
        #show()    
    
    if not CHEAT: 
        print "Computing potential events..."
        start_time = time()
        potential_events_llr, potential_events = unzip(compute_potential_events(blips, stats))
        print "...done (%ss)" % (time() - start_time,)
        print "\tTotal blips: %s\n\tBig blips: %s\n\tStation pairs: %s\n" \
            % (stats["compute_pot_events.total_blips"], stats["compute_pot_events.big_blips"], stats["compute_pot_events.station_pairs"])
        print "\tPotential events proposed: %s\n\tPotential events in range: %s\n\tAre likely: %s\n\tPotential events accepted: %s\n" \
            % (stats["compute_pot_events.blip_pairs"], \
            stats["compute_pot_events.in_range"], stats["compute_pot_events.are_likely"], len(potential_events))        
        
	if not NO_FIGURES:
	        figure()
        	pylab.title("Potential event llr histogram")
	        pylab.hist(potential_events_llr, bins=100)
    else:
        std_dev_e = (INF_TIME_STD_DEV, INF_TIME_STD_DEV, INF_MAG_STD_DEV)
        print "Cheating (standard deviations %s)" % (std_dev_e)
        potential_events = [(e, std_dev, SPACE) for e in true_events]
    
    
    if not NO_FIGURES and False:
        # Just detections 
        fig = figure()
        
        sp = pylab.subplot(1,1,1)    
        #sp.set_title("Detections")        
        plot_world(sp, [], {}, blips)
        
        # True state 
        figure()
        sp = pylab.subplot(1,1,1)
        #sp.set_title("True world state")        
        plot_world(sp, true_events, true_assoc, blips)    
        
        # Detections plus event samples 
        figure()
        sp = pylab.subplot(1,1,1)
        #sp.set_title("Detections with event samples")
        for e_m, e_s, x_range in potential_events:
            dt = e_s[0]*2
            dx = e_s[1]*2
            t1 = e_m[0] - dt
            t2 = e_m[0] + dt
            x1 = e_m[1] - dx
            x2 = e_m[1] + dx
            sp.plot([t1, t2, t2, t1, t1], [x1, x1, x2, x2, x1], c=[0,0,1], linewidth=1.2, zorder=10)
        
        plot_world(sp, {}, {}, blips, show_events=False)
                
        # True state plus event samples 
        figure()
        sp = pylab.subplot(1,1,1)
        #sp.set_title("Potential events")
        for e_m, e_s, x_range in potential_events:
            dt = e_s[0]*2
            dx = e_s[1]*2
            t1 = e_m[0] - dt
            t2 = e_m[0] + dt
            x1 = e_m[1] - dx
            x2 = e_m[1] + dx
            sp.plot([t1, t2, t2, t1, t1], [x1, x1, x2, x2, x1], c=[0,0,1], linewidth=1.2, zorder=10)
        
        plot_world(sp, true_events, true_assoc, blips, show_events=True)
        
    
    
        if not GRAPHS_LATER: show()    

        #for i, pe in enumerate(potential_events):
            #e = pe[0]
            #llr = [max([blip_d(b, e) - noise_d(b) for b in detector_blips]) for detector_blips in blips]
            #print "PEvent %s: %s blip_llr: %.4g=%s prior_prob: %.4g std_dev=%s xran=%s" \
                #% (i, print_event(e),  sum(llr), print_flist(llr), event_d(e), print_flist(pe[1]), print_flist(pe[2]))
    

    
    #
    # These don't look right -- it doesn't catch the correct ones?
    #
    #
    #
      # Observation:  in potential events we have a lot of /whopper/ events:  for two incompatible blips, we suppose an event of large size quite far away...... and even before in time?  Ok, a lot of these have very low posterior probability, we should filter by that.

  
    events = []
    assoc = {}
    invassoc = {}
    stats = DefaultDict(default=lambda x: 0.0)
    
    if DEBUG: print "World (prob=%s):\n\tevents = %s\n\tinvassoc = %s" \
                % (world_d(events, assoc, blips), ", ".join([print_event(e) for e in events]), print_invassoc(invassoc, events, blips))

    iterations = ITERATIONS
    stride = STRIDE
    world_ps = [None]*iterations
    event_nums = [None]*iterations
    
    history = []
    full_history = [(-1, true_events, true_assoc)]
    mcmc_scans = []
    
    true_invassoc = invert_dict(true_assoc)
    
    print "MCMC inference for %s iterations\n" % iterations,
    print "True world (world_logp=%s events=%s prec=%.4f recall=%.4f f1=%.4f precb=%.4f recallb=%.4f f1b=%.4f)" % \
        (world_d(true_events, true_assoc, blips), len(true_events), precision(true_events, true_events, true_invassoc), \
        recall(true_events, true_events, true_invassoc), f1(true_events, true_events, true_invassoc), \
        precisionb(true_assoc, true_assoc), recallb(true_assoc, true_assoc), f1b(true_assoc, true_assoc))
    start_time = time()
    local_time = time()
    for i in range(iterations):
        world_logp = world_d(events, assoc, blips)                
        if i % stride == 0: 
            print "Iteration %i (world_logp=%s events=%s prec=%.4f recall=%.4f f1=%.4f precb=%.4f recallb=%.4f f1b=%.4f; time=%ss)" \
                % (i, world_logp, len(events), precision(events, true_events, true_invassoc), recall(events, true_events, true_invassoc), \
                  f1(events, true_events, true_invassoc), precisionb(assoc, true_assoc), recallb(assoc, true_assoc), \
                  f1b(assoc, true_assoc), time()-local_time)
            history.append((i, events[:], assoc.copy()))
            local_time = time()
            
            if VERBOSE:
                inspect_w(events, assoc, blips)
                print stats
                print
        full_history.append((i, events[:], assoc.copy()))
            
        mcmc_scan(events, assoc, invassoc, blips, stats)
        stats["world-tot-logp"] += world_logp 
        stats["world-tot-logp-square"] += world_logp **2
        stats["iterations"] += 1
        
        world_ps[i] = world_logp
        event_nums[i] = len(events)
        
        if DEBUG:
            if DEBUG: print "World (prob=%s):\n\tevents = %s\n\tinvassoc = %s"  \
                    % (world_logp, ", ".join([print_event(e) for e in events]), print_invassoc(invassoc, events, blips))
            print stats

    print "Final iteration %i (world_logp=%s events=%s prec=%.4f recall=%.4f f1=%.4f precb=%.4f recallb=%.4f f1b=%.4f; time=%ss)" \
                % (iterations, world_d(events, assoc, blips) , len(events), precision(events, true_events, true_invassoc), \
                  recall(events, true_events, true_invassoc), f1(events, true_events, true_invassoc), \
                  precisionb(assoc, true_assoc), recallb(assoc, true_assoc), f1b(assoc, true_assoc), time()-local_time)
    history.append((iterations, events[:], assoc.copy()))
    full_history.append((iterations, events[:], assoc.copy()))
    
    print "Inference complete (%ss)" % (time() - start_time,)
    print        
    print "MH moves: %.3g accepted (%.0f of %.0f):" \
            % (safe_div(stats["mh-moves-accepted"], stats["mh-moves"]), stats["mh-moves-accepted"], stats["mh-moves"])
    print "World logprob: average %s, std-dev %s." % (stats["world-tot-logp"]/stats["iterations"],     # bogus stats
                                    sqrt((stats["world-tot-logp-square"]/stats["iterations"]) - (stats["world-tot-logp"]/stats["iterations"])**2))
    print "Birth moves: %.3g accepted (%.0f of %.0f):" % (safe_div(stats["birth_accepted"], stats["birth_move"]), stats["birth_accepted"], stats["birth_move"])
    print "Death moves: %.3g accepted (%.0f of %.0f):" % (safe_div(stats["death_accepted"], stats["death_move"]), stats["death_accepted"], stats["death_move"])
    print "Average mh moves acceptance logp:", safe_div(stats["mh-tot-logp"], stats["mh-moves"])
    print
    print "Stats:", stats


    print
    print "MCMC history:\n\t%s\n" % "\n\t".join(["%s. %s\t%s" % (i, world_d(e,a,blips), s) \
            for ((i,e,a),s) in zip(full_history, ["true world", "initial state"] + mcmc_scans)]),

    output_prefix = "/home/nickjhay/2009-spring-data/ctbt-figs/low-accuracy-"


    figures = []  
    
    if not NO_FIGURES:
        print "Plotting figures...",    
        
        # plot true state with potential events
        figures.append(figure())
        sp = pylab.subplot(2,1,1)
        sp.set_title("True world state")
        plot_world(sp, true_events, true_assoc, blips)
            
        pevents = [e[0] for e in potential_events]
    
        sp = pylab.subplot(2,1,2)
        sp.set_title("Potential events")
#        sp.scatter([e[0] for e in pevents], [e[1] for e in pevents], c=[(0, 0, 1) for e in pevents], s=[2*e[2]**2 for e in pevents], 
#            linewidths=[FIG_SCALE * 0.2 for e in pevents], marker='s', alpha=0)
            
        for e_m, e_s, x_range in potential_events:
            dt = e_s[0]*2
            dx = e_s[1]*2
            t1 = e_m[0] - dt
            t2 = e_m[0] + dt
            x1 = e_m[1] - dx
            x2 = e_m[1] + dx
            sp.plot([t1, t2, t2, t1, t1], [x1, x1, x2, x2, x1], c=[0,0,1], linewidth=1.2, zorder=10)

        plot_world(sp, true_events, true_assoc, blips, show_events=True)
        
    
    
        # plot final state
        true_events.sort()
        events.sort()    
        
        figures.append(figure())
        sp = pylab.subplot(2,1,1)
        sp.set_title("Final world state (iteration %s)" % (iterations,))
        plot_world(sp, events, assoc, blips)
        print
        print
        sp = pylab.subplot(2,1,2)
        sp.set_title("True world state")
        plot_world(sp, true_events, true_assoc, blips)
    
    
        #plot history
        for i, (iteration, h_events, h_assoc) in enumerate(history):
            if i % 3 == 0:
                figures.append(figure())
            sp = pylab.subplot(3, 1, i % 3+1)
            sp.set_title("State at iteration %s" % iteration,)
            plot_world(sp, h_events, h_assoc, blips)    
    
        print "done"
    
        # save figures to disk
        #print "Saving %s figures with prefix \"%s\"..." % (len(figures), output_prefix),        
    
        #for i, fig in enumerate(figures):
            #fig.savefig("%s%02i.eps" % (output_prefix, i), format="eps", dpi=4000)
        
        #print "done"
        
        show()





def unique(lst, pred):
    out = None
    for x in lst:
        if pred(x):
            if out != None:
                raise ValueError("Unique: found twice")
            out = x
    if out == None:
        raise ValueError("Unique: failed to find")
    return out







# ============================
# Useful data inspection tools
# ============================

def iw(idx):
    h_i, h_events, h_assoc = unique(full_history, lambda x: x[0] == idx)
    inspect_w(h_events, h_assoc, blips)


#import pylab
#from pylab import show, subplot

def pw(idx):
    import pylab
    from pylab import show, subplot
    h_i, h_events, h_assoc = unique(full_history, lambda x: x[0] == idx)
    pylab.figure()
    sp = pylab.subplot(111)
    sp.set_title("Iteration %s" % idx,)
    plot_world(sp, h_events, h_assoc, blips)

#import pdb
#pm = pdb.pm



def set_parameters(options):
    global EVENT_INTENSITY, TOTAL_EVENT_INTENSITY, NOISE_INTENSITY, TOTAL_NOISE_INTENSITY, NOISE_MAG_SCALE, TIME_STD_DEV, MAG_STD_DEV
    global POTENTIAL_EVENT_LLR_THRESHOLD1, POTENTIAL_EVENT_LLR_THRESHOLD2, POTENTIAL_EVENT_MAG_THRESHOLD, FIG_SCALE
    global INF_TIME_STD_DEV, INF_MAG_STD_DEV, MAG_DECAY
    global CHEAT, GRAPHS_LATER, MEAN_ASSOC_LINES, ITERATIONS, STRIDE, VERBOSE, SHOW_DENSITY, NO_FIGURES, SEED
    
    EVENT_INTENSITY = options.event_intensity / SPACE_SIZE
    TOTAL_EVENT_INTENSITY = EVENT_INTENSITY * SPACE_SIZE * TIME_SIZE

    NOISE_INTENSITY = options.noise_intensity                 # per time unit
    TOTAL_NOISE_INTENSITY = NOISE_INTENSITY * TIME_SIZE

    NOISE_MAG_SCALE = options.noise_scale / log(10)
    
    TIME_STD_DEV = options.time_std_dev
    MAG_STD_DEV = options.mag_std_dev
    
    MAG_DECAY = (EVENT_MAG_MIN - NOISE_MAG_MIN - MAG_STD_DEV) / float(SPACE_SIZE)
    
    if options.inf_time_std_dev != None:
        INF_TIME_STD_DEV = options.inf_time_std_dev
    else:
        INF_TIME_STD_DEV = TIME_STD_DEV
    INF_MAG_STD_DEV = MAG_STD_DEV
    
    POTENTIAL_EVENT_LLR_THRESHOLD1 = options.potential_llr_threshold1
    POTENTIAL_EVENT_LLR_THRESHOLD2 = options.potential_llr_threshold2
    POTENTIAL_EVENT_MAG_THRESHOLD = options.potential_mag_threshold
    FIG_SCALE = options.fig_scale
    
    CHEAT = options.cheat
    ITERATIONS = options.iterations
    STRIDE = options.stride
    
    GRAPHS_LATER = options.graphs_later
    MEAN_ASSOC_LINES = options.mean_assoc_lines
    VERBOSE = options.verbose
    SHOW_DENSITY = options.show_density
    NO_FIGURES = options.no_figures
    SEED = options.seed

    
    
def print_parameters():
    #print "Space size: %.4f" % (SPACE_SIZE,)
    #print "Time size: %.4f" % (TIME_SIZE,)
    print "Random seed: %s" % (SEED,)
    print "Event intensity: %.4f / SPACE_SIZE" % (EVENT_INTENSITY * SPACE_SIZE,)
    print "Noise intensity: %.4f" % (NOISE_INTENSITY,)
    print "Noise scale: %.4g / log(10)" % (NOISE_MAG_SCALE * log(10),)
    print "Time standard deviation: %.4g" % (TIME_STD_DEV,)
    print "Magnitude standard deviation: %.4g" % (MAG_STD_DEV,)
    print "Time standard deviation (inference): %.4g" % (INF_TIME_STD_DEV,)
    print "Magnitude standard deviation (inference): %.4g" % (INF_MAG_STD_DEV,)    
    print "Potential event blip magnitude threshold: %.4g" % (POTENTIAL_EVENT_MAG_THRESHOLD,)
    print "Potential event log-likelihood ratio threshold (first): %.4g" % (POTENTIAL_EVENT_LLR_THRESHOLD1,)
    print "Potential event log-likelihood ratio threshold (second): %.4g" % (POTENTIAL_EVENT_LLR_THRESHOLD2,)        
    print "Figure scale: %.4g" % (FIG_SCALE,)
    print "Cheat mode: %s" % (cond("ON", CHEAT, "off"),)
    print "Showing density: %s" % (SHOW_DENSITY,)
    print "Using MCMC moves:\n\t%s\n\n" % "\n\t".join(["%i. %s" % (i,s) for i,s in enumerate(MCMC_MOVES)]),

def parse_options():
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option("-e", "--event-intensity", type="float", default=EVENT_INTENSITY * SPACE_SIZE,
                      help="Event intensity per time unit (default %.4f)." % (EVENT_INTENSITY * SPACE_SIZE),)
    parser.add_option("-n", "--noise-intensity", type="float", default=NOISE_INTENSITY,
                      help="Noise blip intensity per time unit per detector (default %.4f)." % NOISE_INTENSITY,)
    parser.add_option("-s", "--noise-scale", type="float", default=NOISE_MAG_SCALE * log(10),
                      help=("Noise scale for the exponential distribution creating noise blip magnitude (default %.4f). " + \
                           "Multiple of 1/log(10).") % (NOISE_MAG_SCALE * log(10)),)
    parser.add_option("-t", "--time-std-dev", type="float", default=TIME_STD_DEV,
                      help="Standard deviation of event arrival time variation (default %.4f)" % TIME_STD_DEV,)
    parser.add_option("--inf-time-std-dev", type="float", default=None,
                      help="Standard deviation of event arrival time variation used by inference.")    
    parser.add_option("-m", "--mag-std-dev", type="float", default=MAG_STD_DEV,
                      help="Standard deviation of event arrival magnitude variation (default %.4f)" % MAG_STD_DEV,)
    parser.add_option("-p", "--potential-llr-threshold1", type="float", default=POTENTIAL_EVENT_LLR_THRESHOLD1,
                      help="First log-likelihood ratio threshold for potential events (default %.4f)" % POTENTIAL_EVENT_LLR_THRESHOLD1,)
    parser.add_option("-q", "--potential-llr-threshold2", type="float", default=POTENTIAL_EVENT_LLR_THRESHOLD2,
                      help="Second log-likelihood ratio threshold for potential events (default %.4f)" % POTENTIAL_EVENT_LLR_THRESHOLD2,)    
    parser.add_option("-r", "--potential-mag-threshold", type="float", default=POTENTIAL_EVENT_MAG_THRESHOLD,
                      help="Minimum magnitude threshold for blips used to find potential events (default %.4f)" % POTENTIAL_EVENT_MAG_THRESHOLD,)    
    
    parser.add_option("-i", "--iterations", type="int", default = ITERATIONS,
                      help="Iterations to run inference for (default %s)." % ITERATIONS,)
    parser.add_option("-d", "--stride", type="int", default = STRIDE,
                      help="Iterval at which to output status updates (default %s)." % STRIDE,)
    parser.add_option("-c", "--cheat", action="store_true", dest="cheat", default=CHEAT,
                      help="Activate cheat mode (proposes from the actual events)")
    
    parser.add_option("-f", "--fig-scale", type="float", default=FIG_SCALE,
                      help="Figure scale (default %.4f)" % FIG_SCALE,)   
    parser.add_option("-g", action="store_true", dest="graphs_later", default=GRAPHS_LATER,
                      help="Display all graphs later.")    
    parser.add_option("-a", action="store_false", dest="mean_assoc_lines", default=MEAN_ASSOC_LINES,
                      help="Use association lines rather than mean travel time lines.")
    parser.add_option("-y", "--show-density", action="store_true", dest="show_density", default=SHOW_DENSITY,
                      help="Show little probability densities.")                      
    parser.add_option("-v", action="store_true", dest="verbose", default=VERBOSE,
                      help="Verbose.")
    parser.add_option("-x", "--no-figures", action="store_true", dest="no_figures", default=NO_FIGURES,
                      help="No figures.")
    parser.add_option("--seed", type="int", default=SEED, help="Set random seed.")
    
    (options, args) = parser.parse_args()
    
    set_parameters(options)
    


# ==========
# Parameters
# ==========
# Using natural units:  velocity = 1.

SPACE = [0, 100]
SPACE_SIZE = SPACE[1] - SPACE[0]
TIME = [0, 1000]
TIME_SIZE = TIME[1] - TIME[0]

TIME_EXTENSION = SPACE_SIZE            # want to create noise whilst waiting for wavefronts to propagation after events magically stop

EVENT_INTENSITY = 0.01 / SPACE_SIZE    # per spacetime unit
NOISE_INTENSITY = 0.5                  # per time unit;   try 1.0 too
TOTAL_EVENT_INTENSITY = EVENT_INTENSITY * SPACE_SIZE * TIME_SIZE
TOTAL_NOISE_INTENSITY = NOISE_INTENSITY * TIME_SIZE

EVENT_MAG_MIN = 3
NOISE_MAG_MIN = 0
NOISE_MAG_SCALE = 1/log(10)


# hacky:  the densities below use the INF_* versions, since generation only uses the sampler, inference only uses the density.
#         set to something small, like 0.000001, to effectively remove the random element.
TIME_STD_DEV = 2.0 # was 2.0
MAG_STD_DEV = .1  # was .3
#INF_TIME_STD_DEV = 2.0 # was 2.0
#INF_MAG_STD_DEV = .3  # was .3

#TIME_STD_DEV = 0.001
#MAG_STD_DEV = 0.001
INF_TIME_STD_DEV = TIME_STD_DEV
INF_MAG_STD_DEV = MAG_STD_DEV
#INF_TIME_STD_DEV = 100*TIME_STD_DEV
#INF_MAG_STD_DEV = 100*MAG_STD_DEV


# per unit time == unit distance;  this is loss simply due to energy loss: the wave front is 0-D.
MAG_DECAY = (EVENT_MAG_MIN - NOISE_MAG_MIN - MAG_STD_DEV) / float(SPACE_SIZE)

# FIX ME:  the above is a hack; too many blips, we eventually get arrival blips which are negative, which breaks the noise model.

NUM_DETECTORS = 5
DETECTORS = [0.0, 30.0, 40.0, 80.0, 100.0]      # This must be sorted

INDEPENDENCE_SAMPLER_PROB = 0.2
DEATH_PROB = 0.5
#POTENTIAL_EVENT_LLR_THRESHOLD = -10.0           # Log-likelihood ratio threshold
POTENTIAL_EVENT_LLR_THRESHOLD1 = 10
POTENTIAL_EVENT_LLR_THRESHOLD2 = 20
POTENTIAL_EVENT_MAG_THRESHOLD = 0.5

FIG_SCALE = 1

CHEAT = False
GRAPHS_LATER = False
MEAN_ASSOC_LINES = True
SHOW_DENSITY = False

ITERATIONS = 200
STRIDE = 10
VERBOSE = False
NO_FIGURES = False

SEED = 137

if __name__ == "__main__":
    try:
        parse_options()
        print_parameters()
        random.seed(SEED)
        if not NO_FIGURES:
            import pylab
            from pylab import show, subplot
            print "Pylab imported"
        main()
        
    except SystemExit:
        raise
    except:
        import pdb, traceback, sys
        traceback.print_exc(file=sys.stdout)
        pdb.post_mortem(sys.exc_traceback)
        raise
