#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 16:31:22 2022

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

####============================ description ==============================####

=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================

####=======================================================================####
"""
import configargparse
import platform #platform模块给我们提供了很多方法去获取操作系统的信息 
import subprocess
import torch
import re
import os

class CheckSRU(configargparse.Action):
    """
    # For command-line option parsing
    """
    def __init__(self, option_strings, dest, **kwargs):
        super(CheckSRU, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values == 'SRU':
            check_sru_requirement(abort=True)
        # Check pass, set the args.
        setattr(namespace, self.dest, values)

def check_sru_requirement(abort=False):
    """
    # This SRU version implements its own cuda-level optimization,
    # so it requires that:
    # 1. `cupy` and `pynvrtc` python package installed.
    # 2. pytorch is built with cuda support.
    # 3. library path set: export LD_LIBRARY_PATH=<cuda lib path>.
    
    Return True if check pass; if check fails and abort is True,
    raise an Exception, othereise return False.
    """

    # Check 1.
    try:
        if platform.system() == 'Windows':
            subprocess.check_output('pip freeze | findstr cupy', shell=True)
            subprocess.check_output('pip freeze | findstr pynvrtc',
                                    shell=True)
        else:  # Unix-like systems
            subprocess.check_output('pip freeze | grep -w cupy', shell=True)
            subprocess.check_output('pip freeze | grep -w pynvrtc',
                                    shell=True)
    except subprocess.CalledProcessError:
        if not abort:
            return False
        raise AssertionError("Using SRU requires 'cupy' and 'pynvrtc' "
                             "python packages installed.")

    # Check 2.
    if torch.cuda.is_available() is False:
        if not abort:
            return False
        raise AssertionError("Using SRU requires pytorch built with cuda.")

    # Check 3.
    pattern = re.compile(".*cuda/lib.*")
    ld_path = os.getenv('LD_LIBRARY_PATH', "")
    if re.match(pattern, ld_path) is None:
        if not abort:
            return False
        raise AssertionError("Using SRU requires setting cuda lib path, e.g. "
                             "export LD_LIBRARY_PATH=/usr/local/cuda/lib64.")

    return True
