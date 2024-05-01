# Run black on .py files
black 1-unet_solution.py 2-semantic_solution.py 3-instance_solution.py

# Convert .py to ipynb
# "cell_metadata_filter": "all" preserve cell tags including our solution tags
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 1-unet_solution.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 2-semantic_solution.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' 3-instance_solution.py

# Create the exercise notebook by removing cell outputs and deleting cells tagged with "solution"
# There is a bug in the nbconvert cli so we need to use the python API instead
python convert-solution.py 1-unet_solution.ipynb 1-unet_exercise.ipynb
python convert-solution.py 2-semantic_solution.ipynb 2-semantic_exercise.ipynb
python convert-solution.py 3-instance_solution.ipynb 3-instance_exercise.ipynb
