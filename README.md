# JTE-CFlow for Low-light Image Enhancement


## Introduction
JTE-CFlow is a novel flow-based generative method for low-light image enhancement, which consists of a joint-attention transformer based conditional encoder (JTE) and a map-wise cross affine coupling flow (CFlow). **Experiments show that JTE-CFlow outperforms current SOTA methods on 7 mainstream low-light datasets with the same architecture**. 

### Evaluation Metrics

<link rel="stylesheet" type="text/css" href="styles.css">

<table style="width: 80%; font-size: 10px;">
    <tr> 
        <th rowspan="2" class="top-bordered right-bordered">Methods </th>
        <th colspan="2" class="top-bordered right-bordered">LOL-v2-real</th>
        <th colspan="2" class="top-bordered right-bordered">LOL-v2-synthetic</th>
        <th colspan="2" class="top-bordered right-bordered">MIT</th>
        <th colspan="2" class="top-bordered right-bordered">SID</th>
        <th colspan="2" class="top-bordered right-bordered">SMID</th>
        <th colspan="2" class="top-bordered right-bordered">SDSD-indoor</th>
        <th colspan="2" class="top-bordered right-bordered">SDSD-outdoor</th>
    </tr>
    <tr>
        <th> PSNR </th> 
        <th class="right-bordered"> SSIM </th>
        <th> PSNR </th>
        <th class="right-bordered"> SSIM </th>
        <th> PSNR </th>
        <th class="right-bordered"> SSIM </th>
        <th> PSNR </th>
        <th class="right-bordered"> SSIM </th>
        <th> PSNR </th>
        <th class="right-bordered"> SSIM </th>
        <th> PSNR </th>
        <th class="right-bordered"> SSIM </th>
        <th> PSNR </th>
        <th class="right-bordered"> SSIM </th>
    </tr>
    <tr>
        <td class="right-bordered"> MIRNet </td>
        <td> 20.02 </td>
        <td class="right-bordered"> 0.820 </td>
        <td> 21.94 </td>
        <td class="right-bordered"> 0.876 </td>
        <td> 23.73 </td>
        <td class="right-bordered"> 0.925 </td>
        <td> 20.84 </td>
        <td class="right-bordered"> 0.605 </td>
        <td> 25.66 </td>
        <td class="right-bordered"> 0.762 </td>
        <td> 24.38 </td>
        <td class="right-bordered"> 0.864 </td>
        <td> 27.13 </td>
        <td class="right-bordered"> 0.837 </td>
    </tr>
    <tr>
        <td class="right-bordered"> SNR </td>
        <td> 21.48 </td>
        <td class="right-bordered"> 0.849 </td>
        <td> 24.14 </td>
        <td class="right-bordered"> 0.928 </td>
        <td> 24.90 </td>
        <td class="right-bordered"> 0.901 </td>
        <td> 22.867 </td>
        <td class="right-bordered"> 0.625 </td>
        <td> 28.49 </td>
        <td class="right-bordered"> 0.805 </td>
        <td> 29.44 </td>
        <td class="right-bordered"> 0.894 </td>
        <td> 28.66 </td>
        <td class="right-bordered"> 0.866 </td>
    </tr>
    <tr>
        <td class="right-bordered"> LLFormer </td>
        <td> 20.99 </td>
        <td class="right-bordered"> 0.811 </td>
        <td> 23.74 </td>
        <td class="right-bordered"> 0.902 </td>
        <td> 25.75 </td>
        <td class="right-bordered"> 0.923 </td>
        <td> 21.26 </td>
        <td class="right-bordered"> 0.575 </td>
        <td> 27.92 </td>
        <td class="right-bordered"> 0.785 </td>
        <td> 29.65 </td>
        <td class="right-bordered"> 0.874 </td>
        <td> 28.73 </td>
        <td class="right-bordered"> 0.838 </td>
    </tr>
    <tr>
        <td class="right-bordered"> SMG </td>
        <td> 24.03 </td>
        <td class="right-bordered"> 0.820 </td>
        <td> 24.98 </td>
        <td class="right-bordered"> 0.893 </td>
        <td> 25.23 </td>
        <td class="right-bordered"> 0.854 </td>
        <td> 22.70 </td>
        <td class="right-bordered"> 0.556 </td>
        <td> 26.97 </td>
        <td class="right-bordered"> 0.725 </td>
        <td> 26.89 </td>
        <td class="right-bordered"> 0.802 </td>
        <td> 26.33 </td>
        <td class="right-bordered"> 0.809 </td>
    </tr>
    <tr>
        <td class="right-bordered"> LLFlow </td>
        <td> 19.67 </td>
        <td class="right-bordered"> 0.852 </td>
        <td> 23.43 </td>
        <td class="right-bordered"> 0.933 </td>
        <td> 24.70 </td>
        <td class="right-bordered"> 0.925 </td>
        <td> 19.39 </td>
        <td class="right-bordered"> 0.615 </td>
        <td> 27.45 </td>
        <td class="right-bordered"> 0.804 </td>
        <td> 25.46 </td>
        <td class="right-bordered"> 0.896 </td>
        <td> 28.82 </td>
        <td class="right-bordered"> 0.869 </td>
    </tr>
    <tr class="bottom-bordered bold-top-border">
        <td class="right-bordered "> <b>JTE-CFlow (Ours)<b> </td>
        <td> <b>24.06<b> </td>
        <td class="right-bordered"> <b>0.878<b> </td>
        <td> <b>25.21<b> </td>
        <td class="right-bordered"> <b>0.942<b> </td>
        <td> <b>25.90<b> </td>
        <td class="right-bordered"> <b>0.927<b> </td>
        <td> <b>22.869<b> </td>
        <td class="right-bordered"> <b>0.644<b> </td>
        <td> <b>28.68<b> </td>
        <td class="right-bordered"> <b>0.810<b> </td>
        <td> <b>30.39<b> </td>
        <td class="right-bordered"> <b>0.908<b> </td>
        <td> <b>30.75<b> </td>
        <td class="right-bordered"> <b>0.885<b> </td>
    </tr>

</table>
(We remove the GT correction operation when obtaining the metrics of LLFlow for fair comparison. The enhanced image of SMG is with the size 512×512×3 that is different from the original ground truth (GT) image of each testing set, we rescale the enhanced images of SMG to have the same size with the original GT image of each testing set for fair comparison.)


### Visual Quality
<img src="./figure/visual-quality.png" width="800"/>



## Dataset

- LOLv2 (Real & Synthetic): Please refer to the papaer [[From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (CVPR 2020)]](https://github.com/flyywh/CVPR-2020-Semi-Low-Light).

- MIT: Please refer to the paper [[Learning Enriched Features for Real Image Restoration and Enhancement (ECCV 2020)]](https://github.com/swz30/MIRNet).

- SID & SMID & SDSD (indoor & outdoor): Please refer to the paper [[SNR-aware Low-Light Image Enhancement (CVPR 2022)]](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance).




## Testing

### Pre-trained Models

Please download our pre-trained models via the following links [[Baiduyun (extracted code:  u09y)]](https://pan.baidu.com/s/1GPDPEzxYzEqGo0-oig97iw?pwd=u09y) [[Google Drive]](https://drive.google.com/drive/folders/15SDM0rctkjxn9lgdBmvSNEwpUABYW7Sf?usp=drive_link).

### Run the testing code 

You can test the model with paired data and obtain the evaluation metrics. You need to specify the data path ```dataroot_LR```, ```dataroot_GT```, and model path ```model_path``` in the config file. Then run
```bash
python test.py
```


## Acknowledgments
Our code is based on [LLFlow](https://github.com/wyf0912/LLFlow), [SNR](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance).

## Contact
If you have any questions, please feel free to contact the authors via [hynjupt@gmail.com](hynjupt@gmail.com).
