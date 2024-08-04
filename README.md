#  ZMAR-SNFlow:Restoration for low-light images with massive zero-element pixels


## Introduction
ZMAR-SNFlow is a novel flow-based generative method  to restore low-light images with massive zero-element pixels., which consists of  a zero-element mask attention based Restormer
 (ZMAR) encoder and a strengthened normalizing flow (SNFlow). **Experiments show that ZMAR-SNFlow outperforms current SOTA methods on 6 mainstream low-light datasets with the same architecture**. 

### Evaluation Metrics

<link rel="stylesheet" type="text/css" href="styles.css">

<table style="width: 80%; font-size: 10px;">
    <tr> 
        <th rowspan="2" class="top-bordered right-bordered">Methods </th>
        <th colspan="2" class="top-bordered right-bordered">LOL-v1</th>
        <th colspan="2" class="top-bordered right-bordered">LOL-v2-real</th>
        <th colspan="2" class="top-bordered right-bordered">LOL-v2-synthetic</th>
        <th colspan="2" class="top-bordered right-bordered">SDSD-indoor</th>
        <th colspan="2" class="top-bordered right-bordered">SDSD-outdoor</th>
        <th colspan="2" class="top-bordered right-bordered">SID</th>
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
    </tr>
    <tr>
        <td class="right-bordered"> MIRNet </td>
        <td> 20.61 </td>
        <td class="right-bordered"> 0.732 </td>
        <td> 21.17 </td>
        <td class="right-bordered"> 0.757 </td>
        <td> 22.13 </td>
        <td class="right-bordered"> 0.862 </td>
        <td> 28.11 </td>
        <td class="right-bordered"> 0.847 </td>
        <td> 28.83 </td>
        <td class="right-bordered"> 0.858 </td> 
        <td> 20.82 </td>
        <td class="right-bordered"> 0.605 </td> 
    </tr>
    <tr>
        <td class="right-bordered">  Retinexformer </td>
        <td> 23.27 </td>
        <td class="right-bordered"> 0.805 </td>
        <td> 21.29 </td>
        <td class="right-bordered"> 0.802 </td>
        <td> 24.73 </td>
        <td class="right-bordered"> 0.901 </td>
        <td> 30.08 </td>
        <td class="right-bordered"> 0.893 </td>
        <td> 28.85 </td>
        <td class="right-bordered"> 0.853 </td>
        <td> 21.75 </td>
        <td class="right-bordered"> 0.618 </td>
    </tr>
    <tr>
        <td class="right-bordered"> LLFormer </td>
        <td> 20.37 </td>
        <td class="right-bordered"> 0.677 </td>
        <td> 18.95 </td>
        <td class="right-bordered"> 0.697 </td>
        <td> 22.67 </td>
        <td class="right-bordered"> 0.814 </td>
        <td> 28.05 </td>
        <td class="right-bordered"> 0.837 </td>
        <td> 28.72 </td>
        <td class="right-bordered"> 0.849 </td>
        <td> 20.91 </td>
        <td class="right-bordered"> 0.582 </td>        
    </tr>
    <tr>
        <td class="right-bordered">  Restormer </td>
        <td> 21.84 </td>
        <td class="right-bordered"> 0.794 </td>
        <td> 21.70 </td>
        <td class="right-bordered"> 0.794 </td>
        <td> 24.02 </td>
        <td class="right-bordered"> 0.902 </td>
        <td> 29.15 </td>
        <td class="right-bordered"> 0.869 </td>
        <td> 27.56 </td>
        <td class="right-bordered"> 0.835 </td>
        <td> 21.16 </td>
        <td class="right-bordered"> 0.637 </td> 
    </tr>
    <tr>
        <td class="right-bordered"> LLFlow </td>
        <td> 20.09 </td>
        <td class="right-bordered"> 0.829 </td>
        <td> 19.19 </td>
        <td class="right-bordered"> 0.823 </td>
        <td> 22.19 </td>
        <td class="right-bordered"> 0.901 </td>
        <td> 27.45</td>
        <td class="right-bordered"> 0.899 </td>
        <td> 28.90 </td>
        <td class="right-bordered"> 0.869 </td>
        <td> 18.63 </td>
        <td class="right-bordered"> 0.609 </td>       
    </tr>
    <tr class="bottom-bordered bold-top-border">
        <td class="right-bordered "> <b>our ZMAR-SNFlow<b> </td>
        <td> <b>23.37<b> </td>
        <td class="right-bordered"> <b>0.843<b> </td>
        <td> <b>24.15<b> </td>
        <td class="right-bordered"> <b>0.862<b> </td>
        <td> <b>25.62<b> </td>
        <td class="right-bordered"> <b>0.928<b> </td>
        <td> <b>30.60<b> </td>
        <td class="right-bordered"> <b>0.905<b> </td>
        <td> <b>30.03<b> </td>
        <td class="right-bordered"> <b>0.876<b> </td>
        <td> <b>21.80<b> </td>
        <td class="right-bordered"> <b>0.638<b> </td>    
    </tr>

</table>
( For fair comparsion, all methods are re-trained and tested on each dataset, and the training and testing low-light images with our zero-element mask set. We remove the GT correction operation when obtaining the metrics of LLFlow for fair comparison.)


### Visual Quality
![image](https://github.com/user-attachments/assets/e99e7cef-645e-45f5-9572-d9106543f3b9)





## Dataset

- LOLv2 (Real & Synthetic): Please refer to the papaer [[From Fidelity to Perceptual Quality: A Semi-Supervised Approach for Low-Light Image Enhancement (CVPR 2020)]](https://github.com/flyywh/CVPR-2020-Semi-Low-Light).

- SID & SDSD (indoor & outdoor): Please refer to the paper [[SNR-aware Low-Light Image Enhancement (CVPR 2022)]](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance).




## Testing

### Pre-trained Models

Please download our pre-trained models via the following links [[Baiduyun (extracted code:ZMAR)]](https://pan.baidu.com/s/1snS9TcNhav1nYnjTeAUpoA?pwd=ZMAR ) 

### Run the testing code 

You can test the model with paired data and obtain the evaluation metrics. You need to specify the data path ```dataroot_LR```, ```dataroot_GT```, and model path ```model_path``` in the config file. Then run
```bash
python test_LOLv1_v2_real.py
```


## Acknowledgments
Our code is based on [LLFlow](https://github.com/wyf0912/LLFlow), [Restormer](https://github.com/swz30/Restormer)).

## Contact
If you have any questions, please feel free to contact the authors via [zhangbo_boniu@163.com](zhangbo_boniu@163.com).
