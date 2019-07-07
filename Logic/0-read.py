import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
import nipype.interfaces.utility as util  # utility
import nipype.pipeline.engine as pe  # pypeline engine
import os  # system functions
import numpy as np
import nibabel as nib
from nipype.workflows.dmri.fsl.dti import create_eddy_correct_pipeline,\
create_bedpostx_pipeline
controls = []
for subject in os.listdir("../Data/ixi"):
    print (subject)
    for dir in os.listdir("../Data/ixi/" + subject):
        print('-'+dir)
        try:
            for img in os.listdir("../Data/ixi/" + subject+"/" + dir+ "/PD/NIfTI/"):
                PD = nib.load("../Data/ixi/" + subject+"/" + dir+ "/PD/NIfTI/" + img)
                controls.append(PD)
                print(PD.shape)
        except:
            pass
# from zoo import init_nncontext
# sc = init_nncontext()
