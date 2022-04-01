import os
import json
import random
import argparse
import numpy as np
from mturk_cores import MTurkManager, print_log
from datetime import date
from datetime import datetime
import copy


def load_frontend_setting(html_url):
    html_layout = open(html_url, 'r').read()
    QUESTION_XML = """<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
            <HTMLContent><![CDATA[{}]]></HTMLContent>
            <FrameHeight>800</FrameHeight>
            </HTMLQuestion>"""
    question_xml = QUESTION_XML.format(html_layout)
    return question_xml



def main(args):

    # ===============================
    # step1: Parsing Configurations
    # ===============================
    human_eval_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    html_file_folder = os.path.join(human_eval_root_dir, "user_interface", "human_study")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    worker_result_dir = os.path.join(human_eval_root_dir, "mturk_experiment", "recruit_participants", "MTurk_WorkerGroup_Track.json")

    """
    To test HIT in sandbox, set args.test_hit = True (default);
    If post HIT for production, please set args.test_hit = False.
    """
    config_dir = os.path.join(human_eval_root_dir, "mturk_experiment", "config_sandbox.json") if args.test_hit \
                else os.path.join(human_eval_root_dir, "mturk_experiment", "config.json")
    with open(config_dir, "r") as read_file:
        config = json.load(read_file)
    print_log("INFO", f" ====== Display Your Configuration ====== \n{config}")


    # ===============================
    # step2: MTurkManager Client Setup
    # ===============================
    """
    client functions to view:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/mturk.html#MTurk.Client.create_worker_block
    """
    mturk_manager = MTurkManager(config)
    mturk_manager.set_environment()
    mturk_manager.setup_client()
    client = mturk_manager.client
    print(client.get_account_balance()['AvailableBalance'])



    # ===============================
    # step3: MTurk Tasks  --- Create HIT
    # ===============================
    """Recruited Workers
    """
    with open(worker_result_dir, "r") as read_file:
        recruited_worker_json_file = json.load(read_file)
    
    QualificationTypeIds = {}
    for g, values in recruited_worker_json_file.items():
        QualificationTypeIds[f'WorkerGroup{g}'] = values['qualificationyypeID']


    batch_number, group_number = 20, 10
    """
    Step3: Create HIT with Qualified Workers
    """
    hit_records = {}
    for g in range(group_number):
        QualificationRequirement = []
        QualificationRequirement = copy.deepcopy(config['worker_config']['worker_requirements'])
        QualificationRequirement.append(
                                {
                                "QualificationTypeId": QualificationTypeIds[f'WorkerGroup{g}'],
                                "Comparator": 'Exists',
                                "ActionsGuarded": 'DiscoverPreviewAndAccept'}
                                )
        hit_records[f'WorkerGroup{g}'] = []


        for b in range(batch_number):

            html_file_path = os.path.join(html_file_folder, f"human_eval_batch_{b}_workergroup{g}.html")
            frontend_setting = load_frontend_setting(html_file_path)
            response = mturk_manager.create_per_hit(frontend_setting, QualificationRequirement)
            
            hit_records[f'WorkerGroup{g}'].append({
                "batch": b,
                "hit_id": response['HIT']['HITId']
            })
            

    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H_%M")
    save_results_dir = os.path.join(current_dir, f"human_study_sandbox_{date_time}.json") if args.test_hit \
                    else  os.path.join(current_dir, f"human_study_production_{date_time}.json")
    with open(save_results_dir, 'w') as json_file:
        json.dump(hit_records, json_file, indent = 2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_hit', action='store_true', dest='test_hit', default=True)
    args = parser.parse_args()
    main(args)


