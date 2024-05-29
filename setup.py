import pathlib
import os
import sys
pwd = pathlib.Path(__file__).parent.resolve()
# SET PWD=%~dp0

third_party = os.path.join(pwd, 'third_party')
# SET THIRD_PARTY=%PWD%\third_party
os.environ['TVM_LIBRARY_PATH'] = os.path.join(third_party, 'lib') + ';'+ os.path.join(third_party, 'bin')
# SET TVM_LIBRARY_PATH=%THIRD_PARTY%\lib;%THIRD_PARTY%\bin

os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.join(pwd, 'ops', 'cpp') + ';' + third_party
# SET PATH=%PATH%;%TVM_LIBRARY_PATH%;%PWD%\ops\cpp\;%THIRD_PARTY%
os.environ['PYTORCH_AIE_PATH'] = str(pwd)
# SET PYTORCH_AIE_PATH=%PWD%

sys.path.append(os.environ['TVM_LIBRARY_PATH'])
sys.path.append(third_party)
sys.path.append(os.path.join(pwd, 'ops', 'python'))
sys.path.append(os.path.join(pwd, 'tools'))
# SET PYTHONPATH=%PYTHONPATH%;%TVM_LIBRARY_PATH%;%THIRD_PARTY%
# SET PYTHONPATH=%PYTHONPATH%;%PWD%\ops\python
# @REM SET PYTHONPATH=%PYTHONPATH%;%PWD%\onnx-ops\python
# SET PYTHONPATH=%PYTHONPATH%;%PWD%\tools

os.environ['XRT_PATH'] = os.path.join(third_party, 'xrt-ipu')
# set XRT_PATH=%THIRD_PARTY%\xrt-ipu

os.environ['TARGET_DESIGN'] = ''
os.environ['DEVICE'] = 'phx'
os.environ['XLNX_VART_FIRMWARE'] = os.path.join(pwd, 'xclbin', 'phx')
# set TARGET_DESIGN=
# set DEVICE=phx
# set XLNX_VART_FIRMWARE=%PWD%/xclbin/phx
