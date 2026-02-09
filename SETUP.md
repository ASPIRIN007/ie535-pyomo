\# ie535-pyomo — Setup



\## Windows

conda activate pyomo-course

python -c "import pyomo; print('pyomo ok')"



\## Mac

git clone https://github.com/ASPIRIN007/ie535-pyomo.git

cd ie535-pyomo

conda env create -f environment.yml

conda activate pyomo-course

python -c "import pyomo; print('pyomo ok')"



Note (Mac solvers):

brew install glpk coin-or-cbc ipopt



