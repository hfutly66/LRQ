for g in 1 2 3 4
do
python /luoyan/IntraQ-master/data_generate/mygenerate.py 		\
		--model=resnet18 			\
		--batch_size=256 		\
		--test_batch_size=512 \
		--group=$g \
		--targetPro=0.9 \
		--cosineMargin=0.3 \
		--cosineMargin_upper=0.8 \
		--augMargin=0.5 \
		--save_path_head=/luoyan/IntraQ-master/data_generate/result
		--model_path=/luoyan/IntraQ-master/pytorchcvaa/pretrained/resnet18-0896-77a56f15.pth
done
