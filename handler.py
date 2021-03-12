print('\n\n>>> entering handler')
import os
tmp_dir_list = os.listdir('/tmp/')
print('\n\n>>>> this is tmp_dir_list', tmp_dir_list)
import json

try:
    import unzip_requirements
except ImportError:
    print('Import Error occured')



print('>>> loading google model --', os.listdir('/tmp/'))
from _sentence_transformer.sentence_transformer_Google_queries import TopHeaders
print('>>> done loading the model. Instantiating--', os.listdir('/tmp/'))


top_headers_extractor = TopHeaders(
    model_path='./_sentence_transformer',
    s3_bucket="trained-transformer-models-store",
    file_prefix="sentence_transformer_model_for_word_similariy_originally_fine_tuned_on_google_queries.tar.gz"
    )
print('>>> done Instantiating--', os.listdir('/tmp/'))

def predict_answer(event, context):
    """Provide main prediction API route. Responds to both GET and POST requests."""
    try:
        print('all keys in event:', list(event.keys()))
        body = json.loads(event['body'])
        print('event passed, ', event, '--', os.listdir('/tmp/'))
        top_headers = top_headers_extractor(body['keyword_searched'], body['raw_headers'], body['estimated_frac_of_top_headers'])
        print('predcition', top_headers, '--', os.listdir('/tmp/'))
        return {
                "statusCode": 200,
                "headers": {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    "Access-Control-Allow-Credentials": True

                },
                "body": json.dumps({'answer': top_headers})
            }

    except Exception as e:
        print(repr(e))
        print('Your GET request event should contain "body" with three keys "keyword_searched", "raw_headers", "estimated_frac_of_top_headers". (similar to a curl request)')
        
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error": repr(e)})
        }
