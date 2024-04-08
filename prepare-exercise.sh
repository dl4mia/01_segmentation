# Run black on .py files
black unet_solution.py semantic_solution.py

# Convert .py to ipynb
# "cell_metadata_filter": "all" preserve cell tags including our solution tags
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' unet_solution.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' semantic_solution.py
jupytext --to ipynb --update-metadata '{"jupytext": {"cell_metadata_filter":"all"}}' instance_solution.py

# Create the exercise notebook by removing cell outputs and deleting cells tagged with "solution"
# There is a bug in the nbconvert cli so we need to use the python API instead
python convert-solution.py unet_solution.ipynb unet_exercise.ipynb
python convert-solution.py semantic_solution.ipynb semantic_exercise.ipynb
python convert-solution.py instance_solution.ipynb instance_exercise.ipynb
