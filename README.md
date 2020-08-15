# Deep Plastic Surgery


```diff
- We are cleaning our code to make it more readable. 
- Code is coming soon.
```

<table border="0" width='100%' style="FONT-SIZE:15" >
	 <tr align="center">
		<td width="9.70%" align="left"><img src="./figures/teaser-a.png" alt="" width="99%" ></td>
		<td width="13.90%"><img src="./figures/teaser-b.gif" alt="" width="99%" ></td>
		<td width="20.40%"><img src="./figures/teaser-c.png" alt="" width="99%" ></td>
		<td width="9.70%"><img src="./figures/teaser-d.png" alt="" width="99%" ></td>
		<td width="13.90%"><img src="./figures/teaser-e.gif" alt="" width="99%" ></td>	
		<td width="20.40%" align="right"><img src="./figures/teaser-f.png" alt="" width="99%" ></td>
	 </tr>
 	<tr align="center">
		<td colspan="3">(a) controllable face synthesis</td>
		<td colspan="3">(b) controllable face editing</td>
	</tr>
	 <tr align="center">
		<td colspan="6"><img src="./figures/teaser-g.png" alt="" width="99%" ></td>
	</tr>				 	
	 <tr align="center">
		<td colspan="6">(c) adjusting refinement level <em>l</em></td>
	</tr>	
	</tr>				 	
	 <tr>
		<td colspan="6"><p style="text-align: justify; FONT-SIZE:12">Our framework allows users to (a) synthesize and (b) edit photos based on hand-drawn sketches. (c) Our model works robustly on various sketches by setting refinement level <em>l</em> adaptive to the quality of the input sketches, <em>i.e.</em>, higher <em>l</em> for poorer sketches, thus tolerating the drawing errors and achieving the controllability on sketch faithfulness. Note that our model requires no real sketches for training.</p></td>
	</tr>
</table>


This is a pytorch implementation of the paper.

Shuai Yang, Zhangyang Wang, Jiaying Liu and Zongming Guo.
Deep Plastic Surgery: Robust and Controllable Image Editing with Human-Drawn Sketches, 
accepted by European Conference on Computer Vision (ECCV), 2020.

[[Project]](https://williamyang1991.github.io/projects/ECCV2020) | [[Paper]](https://arxiv.org/abs/2001.02890) | [[Human-Drawn Facial Sketches]](https://williamyang1991.github.io/projects/ECCV2020/DPS/files/human-drawn_facial_sketches.zip)

It is provided for educational/research purpose only. Please consider citing our paper if you find the software useful for your work.

## Usage: 

#### Prerequisites
- Python 2.7
- Pytorch 1.2.0
- matplotlib
- scipy
- Pillow
- torchsample

#### Install
- Clone this repo:
```
git clone https://github.com/TAMU-VITA/DeepPS.git
cd DeepPS/src
```

## Testing Example

- Download pre-trained models from [[Baidu Cloud]](https://pan.baidu.com/s/1QjOWk8Gw4UNN6ajHF8bMjQ)(code:oieu) to `../save/`
- Sketch-to-photo translation with refinment level 1.0
  - setting <i>l</i> to -1 (default) means testing with multiple levels in \[0,1\] with step of l_step (default l_step = 0.25)
  - Results can be found in `../output/`
```
python test.py --l 1.0
```
- Face editing with refinment level 0.0, 0.25, 0.5, 0.75 and 1.0
  - model_task to specify task. SYN for synthesis and EDT for editing
  - specify the task, input image filename, model filename for F and G, respectively
  - Results can be found in `../output/`
```
python test.py --model_task EDT --input_name ../data/EDT/4.png \
--load_F_name ../save/ECCV-EDT-celebaHQ-F256.ckpt --model_name ECCV-EDT-celebaHQ
```
- Use `--help` to view more testing options
```
python test.py --help
```
## Training Examples

- Download pre-trained model F from [[Baidu Cloud]](https://pan.baidu.com/s/1QjOWk8Gw4UNN6ajHF8bMjQ)(code:oieu) to `../save/`
- Prepare your data in `../data/dataset/train/` in form of (I,S):


### Training on image synthesis task
- Train G with default parameters on 256\*256 images
  - Progressively train G64, G128 and G256 on 64\*64, 128\*128 and 256\*256 images like pix2pixHD.
    - step1: for each resolution, G is first trained with a fixed <i>l</i> = 1 to learn the greatest refinement level for 30 epoches (--epoch_pre)
    - step2: we then use <i>l</i> ∈ {i/K}, i=0,...,K where K = 20 (i.e. --max_dilate 21) for 200 epoches (--epoch)
```
python train.py --save_model_name PSGAN-SYN
```
Saved model can be found at `../save/`
- Train G with default parameters on 64\*64 images
  - Prepare your dataset in `../data/dataset64/train/` (for example, provided by [ContextualGAN](https://github.com/elliottwu/sText2Image))
  - Prepare your network F pretrained on 64\*64 images as `../save/ECCV-SYN-celeba-F64.ckpt` 
  - max_level = 1 to indicate only train on level 1 (level 1, 2, 3 means image resolution 64\*64, 128\*128, 256\*256.
  - use_F_level 1 to indicate network F is used on level 1 
  - Specify the max dilation diameter, training level, F model image size
  - AtoB means images are prepared in form of (S,I)
```
python train.py --train_path ../data/dataset64/ \
--max_dilate 9 --max_level 1 --use_F_level 1 \
--load_F_name ../save/ECCV-SYN-celeba-F64.ckpt --img_size 64 \
--save_model_name PSGAN-SYN-64 --AtoB
```

### Training on image editing task
- Train G with default parameters on 256\*256 images
  - Progressively train G64, G128 and G256 on 64\*64, 128\*128 and 256\*256 images like pix2pixHD.
    - step1: for each resolution, G is first trained with a fixed <i>l</i> = 1 to learn the greatest refinement level for 30 epoches (--epoch_pre)
    - step2: we then use <i>l</i> ∈ {i/K}, i=0,...,K where K = 20 (i.e. --max_dilate 21) for 200 epoches (--epoch)
```
python train.py --model_task EDT \
--load_F_name ../save/ECCV-EDT-celebaHQ-F256.ckpt --save_model_name PSGAN-EDT
```
Saved model can be found at `../save/`

- Use `--help` to view more testing options
```
python train.py --help
```
### Contact

Shuai Yang

williamyang@pku.edu.cn

