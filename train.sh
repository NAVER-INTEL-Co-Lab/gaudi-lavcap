# PT_HPU_LAZY_MODE=0 python train.py --cfg-path configs/paper_final.yaml
PT_HPU_LAZY_MODE=0 python gaudi_spawn.py --world_size 8 --use_mpi train.py --cfg-path configs/paper_final.yaml