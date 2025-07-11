# use librealsense 2.51.1 to support both t265 and d435
cmake .. -DBUILD_PYTHON_BINDINGS=true -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=ON -DBUILD_WITH_OPENMP:=ON -DPYTHON_EXECUTABLE=/usr/bin/python3.8
echo "export PATH=$PATH:/usr/local/cuda-10.2/bin" >> ~/.zshrc
echo "export PATH=$PATH:/usr/local/cuda-10.2/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64" >> ~/.zshrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64" >> ~/.bashrc

