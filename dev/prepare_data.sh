source init.sh
rm -r .log; mkdir .log

python local/data_prep.py --input_path=ATLAS_PATH --output_path=ATLAS_OUTPUT  2>&1 | tee .log/data_prep_atlas.log 
python local/data_prep.py --input_path=MR_PATH --output_path=MR_OUTPUT 2>&1 | tee .log/data_prep_mr.log
python local/data_prep.py --input_path=CT_PATH --output_path=CT_OUTPUT 2>&1 | tee .log/data_prep_ct.log
python local/data_prep.py --input_path=MR_TEST2_PATH --output_path=MR_OUTPUT 2>&1 | tee .log/data_prep_mr_test2.log
python local/data_prep.py --input_path=CT_TEST2_PATH --output_path=CT_OUTPUT 2>&1 | tee .log/data_prep_ct_test2.log

