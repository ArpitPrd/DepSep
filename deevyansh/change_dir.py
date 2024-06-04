################# This function moves 2 files together according to the conventions in the BGU_dataset ###############
import os
import shutil
def doit(old_spectral,old_clean,new_spectral,new_clean,n):
    temp=os.getcwd()
    list=[]
    os.chdir(os.path.join(temp,old_spectral))
    for files in os.listdir():
        list.append(files)
    os.chdir(temp)
    dist_path1=os.path.join(os.getcwd(),new_spectral)
    dist_path2=os.path.join(os.getcwd(),new_clean)
    index=0
    for i in list:
        if(index<n):
            source_path1=os.path.join(os.getcwd(),old_spectral,i)
            source_path2=os.path.join(os.getcwd(),old_clean,(i[:-4]+'_clean.png'))
            shutil.move(source_path1,dist_path1)
            shutil.move(source_path2,dist_path2)
            index=index+1
        else:
            return


doit('BGU/data/train_spectral','BGU/data/train_clean','BGU/data/val_spectral','BGU/data/val_clean',1)
