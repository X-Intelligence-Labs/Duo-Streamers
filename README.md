# Duo Streamers: A Streaming Gesture Recognition Framework

<div align="center" style="max-width: 16cm; margin: auto;" markdown="1">

  
**Boxuan Zhu**<sup>1,2</sup> <boxuan.zhu@liverpool.ac.uk>  
**Sicheng Yang**<sup>1</sup> <940863869@qq.com>  
**Zhuo Wang**<sup>1</sup> <zhuohci01@gmail.com>  
**Haining Liang**<sup>3</sup> <hainingliang@hkust-gz.edu.cn>  
**Junxiao Shen**<sup>4,[&#9993;]</sup> <junxiao.shen@bristol.ac.uk>

<br/>

<sup>1</sup> OpenInterX  
<sup>2</sup> University of Liverpool  
<sup>3</sup> HKUST (Guangzhou)  
<sup>4</sup> University of Bristol  

<br/>

</div>

<div align="center" style="margin-top: 20px; margin-bottom: 40px;">
  <!-- “Code” -->
  <a href="https://github.com/JulienInWired/Duo-Streamers"
     style="text-decoration: none;
            display: inline-block;
            margin: 0 10px;
            padding: 8px 16px;
            background: #24292e;  /* GitHub */
            color: #ffffff;
            border-radius: 6px;
            font-weight: bold;">
    <!-- Pic or Font Awesome -->
    Code
  </a>

  <!-- “arXiv” -->
  <a href="https://arxiv.org/abs/xxxx.xxxxx"
     style="text-decoration: none;
            display: inline-block;
            margin: 0 10px;
            padding: 8px 16px;
            background: #b31b1b;  /* arXiv */
            color: #ffffff;
            border-radius: 6px;
            font-weight: bold;">
    <!-- <svg> -->
    arXiv
  </a>
</div>

<div align="center">
  <img src="pics/teaserstreamers.jpg" alt="teaserstreamer" width="700px" />
</div>



## Abstract

Gesture recognition in resource-constrained scenarios faces significant challenges in achieving high accuracy and low latency. The streaming gesture recognition framework, Duo Streamers, proposed in this paper, addresses these challenges through a three-stage sparse recognition mechanism, an RNN-lite model with an external hidden state, and specialized training and post-processing pipelines, thereby making innovative progress in real-time performance and lightweight design. Experimental results show that Duo Streamers matches mainstream methods in accuracy metrics, while reducing the real-time factor by approximately 92.3\%, i.e., delivering a nearly 13-fold speedup. In addition, the framework shrinks parameter counts to 1/38 (idle state) and 1/9 (busy state) compared to mainstream models. In summary, Duo Streamers not only offers an efficient and practical solution for streaming gesture recognition in resource-constrained devices but also lays a solid foundation for extended applications in multimodal and diverse scenarios. Upon acceptance, we will publicly release all models, code, and demos.

<div align="center">
  <video width="640" controls>
    <source src="DuoStreamers_Visualization.mp4" type="video/mp4">
  </video>
</div>

