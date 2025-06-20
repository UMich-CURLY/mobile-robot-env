mkdir -p /usr/lib/dri/
sudo ln -s /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/
sudo ln -s /usr/lib/x86_64-linux-gnu/dri/radeonsi_dri.so /usr/lib/dri/
mv /root/conda/envs/vln/lib/libstdc++.so.6 /root/conda/envs/vln/lib/libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /root/conda/envs/vln/lib/libstdc++.so.6

cp dataset.py /dependencies/habitat-lab/habitat/core/
cp nav.py /dependencies/habitat-lab/habitat/tasks/nav/

pip install supervision==0.14.0
conda install -y pytorch::faiss-gpu
# 4.50 has a bug: https://github.com/huggingface/transformers/issues/36888
pip install transformers==4.49.0
pip install scipy==1.11.2
pip install opencv-python==4.9.0.80