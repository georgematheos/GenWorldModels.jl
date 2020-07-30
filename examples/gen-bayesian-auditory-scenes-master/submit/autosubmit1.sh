#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=mcdermott

echo "Running autosubmit1.sh"
start=$SECONDS

cd /om2/user/mcusi/gen-bayesian-auditory-scenes/
source activate gentle

dataset='vec'
echo "dataset: $dataset"
generatedata=0
echo "generating dataset (1 true, 0 false): $generatedata"
createrec=1
echo "generating tfrecords (1 true, 0 false): $createrec"
neuralnetexpt='paramita'
echo "neural net expt name: $neuralnetexpt"
pretrain=1
echo "pretraining (1 true, 0 false): $pretrain"
usedgx=1
echo "use big gpus for end2end testing: $usedgx"

dreamparallel=40
echo "n_parallel dream scripts (if generating data): $dreamparallel"
ndatapoints=25000
echo "n_datapoints to dream (if generating data): $ndatapoints"

if [ $generatedata -eq 1 ]; then

	echo "Generating from model..."
	#RES should contain the job_id  
	dream_res=$(sbatch --parsable --array=1-$dreamparallel ./submission-scripts/autodream.sh $dataset $dataset $ndatapoints)
	echo "Dreaming ID: ${dream_res}"

	cleanup_res=$(sbatch --parsable --array=1-$dreamparallel --dependency=afterok:${dream_res} ./submission-scripts/autoclean.sh $dataset)
	echo "cleanup ID: ${cleanup_res}"

	createdep="--dependency=afterok:${cleanup_res}"

else

	createdep=""

fi

if [ $createrec -eq 1 ]; then

	cd /om/user/mcusi/dcbasa/
	source ./environ.sh

	ttrain_req="--time=15:00:00"
	ttest_req="--time=10:00:00"

	echo "Making tfrecords..."
	#SBATCH 
	createparallel=15
	chaptersize=`echo "($ndatapoints*($dreamparallel-1))/$createparallel" | bc` #one left for testing
	echo "Creating $createparallel tfrecords with $chaptersize points each!"
	emb_train_res=$(sbatch --parsable ${ttrain_req} --array=1-$createparallel $createdep autocreate.sh $dataset embedding train $chaptersize)
	#test tfrecord should use as much as it can from the dream
	emb_test_res=$(sbatch --parsable ${ttest_req} $createdep autocreate.sh $dataset embedding test $ndatapoints)
	echo "tfrecord- Emb train ID: ${emb_train_res}"
	echo "tfrecord- Emb test ID: ${emb_test_res}"

	est_train_res=$(sbatch --parsable ${ttrain_req} --array=1-$createparallel $createdep autocreate.sh $dataset estimation train $chaptersize)
	est_test_res=$(sbatch --parsable ${ttest_req} $createdep autocreate.sh $dataset estimation test $ndatapoints)
	echo "tfrecord- Est train ID: ${est_train_res}"
	echo "tfrecord- Est test ID: ${est_test_res}"

	end_train_res=$(sbatch --parsable ${ttrain_req} --array=1-$createparallel $createdep autocreate.sh $dataset end2end train $chaptersize)
	end_test_res=$(sbatch --parsable ${ttest_req} $createdep autocreate.sh $dataset end2end test $ndatapoints)
	echo "tfrecord- End train ID: ${end_train_res}"
	echo "tfrecord- End test ID: ${end_test_res}"

	demo_res=$(sbatch --parsable --time=00:30:00 $createdep autocreate.sh $dataset end2end demos)
	echo "tfrecord- End demo ID: ${demo_res}"

	emb_dependency="--dependency=afterok:${emb_train_res},${emb_test_res}"
	est_dependency="--dependency=afterok:${est_train_res},${est_test_res}"
	end_dependency=",${end_train_res},${end_test_res}"

else

	cd /om/user/mcusi/dcbasa
	source ./environ.sh
	emb_dependency=""
	est_dependency=""
	end_dependency=""

fi

if [ $pretrain -eq 1 ]; then
	
	time_request="--time=30:00:00"
	gpu_request="--gres=gpu:GEFORCEGTX1080TI:4" 

	#PARAMS FIle must BE Set up BEFOREHAND!
	echo "Starting net pretrain..."
	emb_net_res=$(name=${neuralnetexpt}_emb g-run sbatch --parsable ${time_request} ${gpu_request} ${emb_dependency} autotrain.sh $dataset embedding 0)
	xf=`echo ${emb_net_res} | cut -d" " -f5` #experiment/foldername... 
	emb_folder=`echo ${xf} | cut -d"/" -f2`
	emb_id=`echo ${emb_net_res} | cut -d" " -f6`
	echo "Training embedding Folder: ${emb_folder}"
	echo "Training embedding ID: ${emb_id}"

	time_request="--time=30:00:00"
	gpu_request="--gres=gpu:GEFORCEGTX1080TI:2" 

	est_net_res=$(name=${neuralnetexpt}_est g-run sbatch --parsable ${time_request} ${gpu_request} ${est_dependency} autotrain.sh $dataset estimation 0)
	xf=`echo ${est_net_res} | cut -d" " -f5`
	est_folder=`echo ${xf} | cut -d"/" -f2`
	est_id=`echo ${est_net_res} | cut -d" " -f6`
	echo "Training estimation Folder: ${est_folder}"
	echo "Training estimation ID: ${est_id}"

	submit2_dependency="--dependency=afterany:${emb_id},${est_id}${end_dependency}"

else

	emb_folder=""
	est_folder=""
	submit2_dependency=""

fi

submit_end_res=$(sbatch --parsable ${submit2_dependency} autosubmit2.sh $dataset ${emb_folder} ${est_folder} ${neuralnetexpt} $usedgx)
echo "Submission script for end2end training (autosubmit2): ${submit_end_res}"
echo "You can copy this command:"
echo "more /om/user/mcusi/dcbasa/slurm-${submit_end_res}.out"

echo "Submitted up to pretrain jobs!"

duration=$(( SECONDS - start ))
echo 'autosubmit1- Time taken in seconds:'
echo $duration
