This page presents the code for the design of TMS-Net (Trustworthy Multi-view Segmentation Network, see our paper, TMS-Net: A segmentation network coupled with a run-time quality control method for robust cardiac image segmentation via https://www.sciencedirect.com/science/article/pii/S0010482522011301?casa_token=C2LxOEbkSuAAAAAA:eiloE-3_9XRrt5HLi3ERF7u25s-0fhcuPLET-6S8UJfvwynpfvNEgCm5ke8dW7I3CMmCpB_EVfukxw) and its training. TMS-Net is a trustworthy segmentation network designed for 3D cardiac images. The network processes 2D slices sampled along the axial, sagittal and coronal views of a 3D MRI volume image by an allocated decoder for each view, by generating three segmentation volume masks for a single input volume. This design allows us to measure the confidence of the network on its outputs, with a cosine similarity metric.  We used computerised noise and Rician noise to evaluate how well this metric estimates the quality of a segmentation mask on run-time. 


PrepareMuitiviewImageDataset.ipynb jupyter notebook can be used to generate 2D multiview images(e.g sagittal) dataset from 3D images. MainFunction.ipynb can be used for training and evaluation TMS-Net. Please feel free to send any inqury about the code to fatmatulzehra.uslu@btu.edu.tr



@article{uslu2023tms,
  title={TMS-Net: A segmentation network coupled with a run-time quality control method for robust cardiac image segmentation},
  author={Uslu, Fatmat{\"u}lzehra and Bharath, Anil A},
  journal={Computers in Biology and Medicine},
  volume={152},
  pages={106422},
  year={2023},
  publisher={Elsevier}
}

