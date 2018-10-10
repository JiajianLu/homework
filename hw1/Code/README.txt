Requirements:
Put Behavioral_Cloning.py, Imitation_learning_with_DAgger.py under the same folder of experts and expert_data

For generating table1, mean and standard deviation of return will be printed in the terminal after running BC agent

Run BC agent on Humanoid task
python Behavioral_Cloning.py expert_data/Humanoid-v2.pkl Humanoid-v2 --render \
            --num_rollouts 20

Run BC agent on Hopper task
python Behavioral_Cloning.py expert_data/Hopper-v2.pkl Hopper-v2 --render \
            --num_rollouts 20

For generating figure1
python Behavioral_Cloning_parameter.py expert_data/Humanoid-v2.pkl Humanoid-v2 --render \
            --num_rollouts 20

For generating figure2, run DAgger on Hopper task

Line for expert is hard coded according to the mean of the expert
Line for BC agent is hard coded based on the mean of BC agent trained with 100 rollouts of expert data

python Imitation_learning_with_DAgger.py experts/Hopper-v2.pkl expert_data/Hopper-v2.pkl Hopper-v2 --render     --num_rollouts 20

