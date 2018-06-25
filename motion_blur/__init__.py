import os

folder = 'C:/Users/slugg/Desktop/2018/2018 gthesis/astro_data'
dir_att = os.path.join

for filename in os.listdir(folder):
    if '.fit' in filename:
        whole_st = dir_att(folder,filename)
        os.rename(whole_st ,whole_st+'s')
