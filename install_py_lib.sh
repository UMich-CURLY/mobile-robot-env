# python3 -m pip install opencv-python pygame jsonpickle lcm
# python3.8 -m pip uninstall scipy numpy matplotlib
# python3.8 -m pip install scipy numpy matplotlib

wget https://github.com/tccoin/PyRecastDetour-Sources/raw/refs/heads/main/dist/PyRecastDetour.cpython-310-x86_64-linux-gnu.so
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
sudo mv PyRecastDetour.cpython-310-x86_64-linux-gnu.so $SITE_PACKAGES/
echo "Installed PyRecastDetour to $SITE_PACKAGES"
echo "Testing import"
python3 -c "import PyRecastDetour as m; print(m, 'OK')"