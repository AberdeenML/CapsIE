python eval_color_prediction_caps2.py --experience SIECaps \
--weights-file /path/to/your/model/model.pth  \
--dataset-root /path/to/dataset/3DIEBench/ \
--exp-dir SIECaps/eval \
--root-log-dir SIECaps/eval/logs/ \
--epochs 300 \
--arch resnet18 \
--batch-size 256 \
--lr 0.001 \
--wd 0.00000 \
--equi-dims 512 \
--device cuda:0 \
--mlp 1111-16-16 \
--caps-type SR \
--deep-end 

