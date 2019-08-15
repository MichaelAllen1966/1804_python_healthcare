# If not already installed, install texlive-xetex and pandoc
# sudo apt-get install texlive-xetex pandoc


jupyter nbconvert --to pdf *.ipynb
jupyter nbconvert --to script *.ipynb
jupyter nbconvert --to html *.ipynb
jupyter nbconvert --to latex *.ipynb

