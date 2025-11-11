# FLMIG_algorithm
Community detection in complex networks is an emerging topic in modern science. The most widely used community detection procedure is modularity optimization, which is an
NP-hard problem. To address this problem, several metaheuristics were used to identify the best possible solution within anacceptable computation time. In this paper we present the Fast
Local Move Iterated Greedy Algorithm to define the community structure in complex network. The FLMIG starts from initial solution generate by the greedy constructive heuristic then iter-
atively improve the quality of solution by deploying the random neighbors community in the reconstruction phase for strong diversification, and the fast local move procedure, which provided
strong intensification with reasonable computation time and . The experimental results on artificial and real-world networks prove the competitiveness of the proposed algorithm over the existing
approach.

The cmmd file has the commands to run the algorithm so download the project and open the project then set the commands in the terminel as showing below : 
python3 ** FLMIG.py ** "path of dataset" ** "number of iterations" ** "beta" ** "ground truth if the dataset has it and None if there is no ground truth"**  "number_of_runtime"

Example : python3 **FLMIG.py ** "path of the dataset" ** 100 ** 0.5 ** "path of the ground-truth file" ** 50
