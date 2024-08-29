# IA2C
IA2C applied to HVT. Initial simplifying assumptions:
(1) Agents receive joint observations;
(2) Private observations are noise-free, no belief tracking;
(3) Actions are deterministic

# How to Run On Org Domain

Install requirements using command pip install -r requirements.txt

Ensure all files are in the same Directory

To train, run command python ia2c_hvt_jointobs_det.py
To generate trajectories from trained policies, run python eval_policy_joitobs.py
To visualize the generated trajectories, run python visualizer.py logfile_State_def.csv ./def.pdf
