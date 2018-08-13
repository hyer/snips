# darknet_scripts
This repo contains my auxilary scripts to work with darknet deep learning famework
1. [How to reproduce YOLOv2 anchors for yolo-voc.cfg?](#how-to-reproduce-yolov2-anchors-for-yolo-voccfg)
2. [How to visualize genereted anchors?](#how-to-visualize-genereted-anchors)
3. [Is gen_anchors.py same as YOLOv2 anchor computation?](#is-gen_anchorspy-same-as-yolov2-anchor-computation)
4. [How to get anchors if My input for network is bigger than 416?](#how-to-get-anchors-if-my-input-for-network-is-bigger-than-416)
5. [How to plot YOLO loss](#how-to-plot-yolo-loss)
6. [YOLO and Anchors tutorial](http://christopher5106.github.io/object/detectors/2017/08/10/bounding-box-object-detectors-understanding-yolo.html) 

### How to reproduce YOLOv2 anchors for yolo-voc.cfg?

2. Download The Pascal VOC Data and unpack it to directory `build\darknet\x64\data\voc` will be created dir `build\darknet\x64\data\voc\VOCdevkit\`:
    * http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    * http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
    * http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
    
    2.1 Download file `voc_label.py` to dir `build\darknet\x64\data\voc`: http://pjreddie.com/media/files/voc_label.py

3. Download and install Python for Windows: https://www.python.org/ftp/python/3.5.2/python-3.5.2-amd64.exe

4. Run command: `python build\darknet\x64\data\voc\voc_label.py` (to generate files: 2007_test.txt, 2007_train.txt, 2007_val.txt, 2012_train.txt, 2012_val.txt)

5. Run command: `type 2007_train.txt 2007_val.txt 2012_*.txt > train.txt`

Next, call <br/> 
```cmd
python gen_anchors.py -filelist //path//to//voc//filelist/list//train.txt -output_dir generated_anchors/voc-reproduce -num_clusters 5
```
<br />
You will have anchors5.txt in generated_anchors/voc-reproduce folder. <br />

### How to visualize genereted anchors?
After completing the steps above, execute <br />
```cmd
python visualize_anchors.py -anchor_dir generated_anchors/voc-reproduce 
```
<br />
Inside the generated_anchors/voc-reproduce directory you will have png visualization of the anchors <br />



### Is gen_anchors.py same as YOLOv2 anchor computation?

<h4> Yes, almost. Look at the two visualaziations below:</h4>
<br />
<ul>

<li>
yolo-voc.cfg anchors are provided by the original author
<img src= 'https://github.com/Jumabek/darknet_scripts/blob/master/generated_anchors/voc-original/yolo-voc.png' />
</li>
<br />

<li>
yolo-voc-reproduce.cfg anchors computed by gen_anchors.py 
<img src= 'https://github.com/Jumabek/darknet_scripts/blob/master/generated_anchors/voc-anchors-reproduce/anchors5.png' />
</li>
<br />
</ul>



### How to get anchors if My input for network is bigger than 416?
Simply change the lines here https://github.com/Jumabek/darknet_scripts/blob/master/gen_anchors.py#L17
to your input dimension.
Then compute the anchors.



### How to plot YOLO loss? 
In order to plot a loss, you first need a log of the <i>darknet train</i> command
For example,below command will save the log into log/aggregate-voc-tiny7.log <br /><br />
```cmd
darknet.exe detector train data/aggregate-voc-tiny7.data cfg/aggregate-voc-tiny7.cfg  backup/aggregate-voc-tiny7/aggregate-voc-tiny7_21000.weights >> log/aggregate-voc-tiny7.log -gpus 0,1
```
<br />
<br />
Next, to plot the loss, execute <br />
```
python plot_yolo_log.py \\path\\to\\log\\aggregate-voc-tiny7.log
```
