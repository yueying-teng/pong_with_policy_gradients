# pong_with_policy_gradients


apt-get install python-opengl -y
apt install xvfb -y
pip install pyvirtualdisplay
pip install piglet
pip install -U gym
pip install -U ale-py


wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar -y
mv "ROMS/Video Olympics - Pong Sports (Paddle) (1977) (Atari, Joe Decuir - Sears) (CX2621 - 99806, 6-99806, 49-75104) ~.bin" "ROMS/pong.bin"
ale-import-roms ROMS/


