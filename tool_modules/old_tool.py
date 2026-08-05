# @register_tool(
#     "generate_cif",
#     "Generate a crystal structure CIF file for a given composition (and optional space group) using CrystaLLM. Please given composition in order of increasing electronegativity and of lowest subscripts.",
#     {"composition": "string", "space_group": "string (optional)", "output_path": "string"},
# )
# def generate_cif(composition: str, output_path: str, space_group: str | None = None):
    
#     output_path = os.path.abspath(output_path)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     with tempfile.TemporaryDirectory() as tmpdir:

#         prompt_path = os.path.join(tmpdir, "prompt.txt")

#         # Step 1: create prompt
#         make_prompt_cmd = [
#             CRYSTALLM_PYTHON,
#             os.path.join(CRYSTALLM_DIR, "bin", "make_prompt_file.py"),
#             composition,
#             prompt_path,
#         ]

#         if space_group:
#             make_prompt_cmd.extend(["--spacegroup", space_group])

#         r1 = subprocess.run(
#             make_prompt_cmd,
#             capture_output=True,
#             text=True,
#         )

#         if r1.returncode != 0:
#             return {
#                 "success": False,
#                 "stage": "make_prompt",
#                 "stdout": r1.stdout,
#                 "stderr": r1.stderr,
#             }

#         # Step 2: Sample raw CIF
#         sample_cmd = [
#             CRYSTALLM_PYTHON,
#             os.path.join(CRYSTALLM_DIR, "bin", "sample.py"),
#             f"out_dir={CRYSTALLM_MODEL_DIR}",
#             f"start=FILE:{prompt_path}",
#             "num_samples=1",
#             "target=file",
#         ]

#         r2 = subprocess.run(
#             sample_cmd,
#             capture_output=True,
#             text=True,
#             cwd=tmpdir,      # sample_1.cif will be written here
#         )

#         if r2.returncode != 0:
#             return {
#                 "success": False,
#                 "stage": "sample",
#                 "stdout": r2.stdout,
#                 "stderr": r2.stderr,
#             }

#         # Step 3: Postprocess
#         processed_dir = os.path.join(tmpdir, "processed")

#         post_cmd = [
#             CRYSTALLM_PYTHON,
#             os.path.join(CRYSTALLM_DIR, "bin", "postprocess.py"),
#             tmpdir,
#             processed_dir,
#         ]

#         r3 = subprocess.run(
#             post_cmd,
#             capture_output=True,
#             text=True,
#         )

#         if r3.returncode != 0:
#             return {
#                 "success": False,
#                 "stage": "postprocess",
#                 "stdout": r3.stdout,
#                 "stderr": r3.stderr,
#             }

#         # Step 4: Find processed CIF
#         cifs = [
#             f for f in os.listdir(processed_dir)
#             if f.endswith(".cif")
#         ]

#         if len(cifs) == 0:
#             return {
#                 "success": False,
#                 "stage": "postprocess",
#                 "stderr": "No processed CIF produced.",
#             }

#         src = os.path.join(processed_dir, cifs[0])

#         shutil.move(src, output_path)

#         return {
#             "success": True,
#             "output_path": output_path,
#         }