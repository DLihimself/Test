import datetime
import pandas as pd
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import logging

import simulator.data_simulator as dtsm
from signal_system import SignalSystem


def create_logger():
    logger = logging.getLogger('signalLogger')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('logs/http_server.log')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


signal_system = SignalSystem()
signal_logger = create_logger()

 
class SignalHttpRequestHandler(BaseHTTPRequestHandler):
    def build_data(self, params):
        fmissing = []
        cur_time = datetime.datetime.now()

        signal_data = pd.DataFrame(columns=dtsm.FULL_FEATURE_COL, index=[0])
        for fname in dtsm.FULL_FEATURE_COL:
            if fname == 'createDtm' or fname == 'wavelength':
                continue

            pvalue = params.get(fname, None)
            if pvalue is None:
                fmissing.append(fname)
                continue
            
            if fname in dtsm.FEATURE_COL:
                signal_data[fname] = float(pvalue[0])
            else:
                signal_data[fname] = pvalue[0]

        if len(fmissing) == 0:
            signal_data['createDtm'] = str(pd.to_datetime(cur_time))
            signal_data['wavelength'] = 3.0*1e8/signal_data['frequency']

        return signal_data, fmissing

    def do_GET(self):
        signal_logger.debug(f'Reveive GET request, url path: {self.path}')

        # 解析url
        url_split = urllib.parse.urlsplit(self.path)

        # 获取path
        path = url_split.path

        # 根据path分发请求
        if path == '/train':
            signal_logger.debug('Handle train request [START]')
            signal_system.train_detect_module()
            signal_system.train_repair_module()
            self.send_response(200)
            self.end_headers()
            signal_logger.debug('Handle train request [DONE]')

        elif path == '/detect' or '/repair':
            req_type = path[1:]
            signal_logger.debug(f'Handle {req_type} request [START]')

            # 获取URL参数
            params = urllib.parse.parse_qs(url_split.query)

            data_full, fmissing = self.build_data(params)
            if len(fmissing) > 0:
                err_msg = f'{fmissing} is not provided'
                self.send_response(400, err_msg)
                self.end_headers()
                signal_logger.warning(f'Handle {req_type} request [FAILED]: {err_msg}')
                return
            
            is_valid, error_features = signal_system.check_full_data(data_full)
            if not is_valid:
                err_msg = f'Invalid features: {error_features}, input params: {params}'
                self.send_response(400, err_msg)
                self.end_headers()
                signal_logger.warning(f'Handle {req_type} request [FAILED]: {err_msg}')
                return

            if req_type == 'detect':
                is_fault = signal_system.detect(data_full)

                data = { "is_fault": is_fault }
                response = json.dumps(data)

                self.send_response(200)
                self.end_headers()
                self.wfile.write(response.encode())

            elif req_type == 'repair':
                is_fault, data_impute = signal_system.repair(data_full)

                data = {
                    'is_fault': is_fault,
                    'data_impute': data_impute.iloc[0].to_dict()
                }
                response = json.dumps(data)

                self.send_response(200)
                self.end_headers()
                self.wfile.write(response.encode())

            signal_logger.debug(f'Handle {req_type} request [DONE], response: {response}')


if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, SignalHttpRequestHandler)
    print('Starting server...')
    httpd.serve_forever()
