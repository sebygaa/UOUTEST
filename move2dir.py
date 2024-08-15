import os
import datetime
import shutil

def move(dir_prefix):
    noww = datetime.datetime.now()
    noww_str = noww.strftime("%Y%m%d%H")[2:]
    dirnam = dir_prefix+noww_str
    dir_list = os.listdir()
    #if not dirnam in dir_list:
    #print(os.path.exists(dirnam))
    if not os.path.exists(dirnam):
        os.mkdir(dirnam)
    for nn in dir_list:
        if nn[-4:] == '.pkl':
            print(nn, ' is moved to ',dirnam)
            shutil.move(nn, os.path.join(dirnam,nn))
        if nn[-4:] == '.txt':
            print(nn, ' is moved to ',dirnam)
            shutil.move(nn, os.path.join(dirnam, nn))    
            
#os.listdir()
#move('res_extLan')