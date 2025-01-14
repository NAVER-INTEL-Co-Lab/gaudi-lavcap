docker pull vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest

docker run -v ./workspace:/workspace \
	   --workdir /workspace \
	   -it --runtime=habana \
	   -e HABANA_VISIBLE_DEVICES=all \
	   -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
	   --cap-add=sys_nice \
	   --net=host \
	   --ipc=host vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest