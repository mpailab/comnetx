python run.py --dataset=acm --T 15 --alpha 0.8 --method mod  --gamma 1 --tau 50
python run.py --dataset=acm --T 15 --alpha 0.8 --method sub --gamma 1 --tau 7

python run.py --dataset=dblp --T 10 --alpha 0.9 --method mod --gamma 1 --tau 50
python run.py --dataset=dblp --T 10 --alpha 0.9 --method sub --gamma 1 --tau 7


python run.py --dataset=wiki --T 12 --alpha 1.7 --method mod --gamma 0.9 --tau 50
python run.py --dataset=wiki --T 6 --alpha 1.7 --method sub --gamma 0.9 --tau 7


python run.py --dataset=pubmed --T 175 --alpha 1.8 --method mod --gamma 1 --tau 50
python run.py --dataset=pubmed --T 175 --alpha 1.8 --method sub --gamma 1 --tau 7


python run.py --dataset=arxiv --T 30 --alpha 2.5 --method mod --gamma 1 --tau 50
python run.py --dataset=arxiv --T 30 --alpha 1.4 --method sub --gamma 1 --tau 4


python run.py --dataset=corafull --T 7 --alpha 1.4 --method mod --gamma 1 --tau 100
python run.py --dataset=corafull --T 20 --alpha 0.9 --method sub --gamma 1 --tau 7


python run.py --dataset=citeseer --T 40 --alpha 0.8 --method mod --gamma 0.9 --tau 100
python run.py --dataset=citeseer --T 60 --alpha 0.8 --method sub --gamma 0.9 --tau 7

python run.py --dataset=Amazon_photos --T 9 --alpha 1.5 --method mod --gamma 1 --tau 50
python run.py --dataset=Amazon_photos --T 9 --alpha 1.5 --method sub --gamma 1 --tau 7
