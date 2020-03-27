import time
import cv2
import yaml
import mock


class WebcamFetcherService(object):
    CHANNEL = 0
    LOG_PATH = 'ServicesLog/WebcamFetcherLog.txt'
    CONFIG_PATH = r'ServicesConfiguration/WebcamFetcher.yaml'
    WAITING_TIME_BETWEEN_ATTEMPTS = "Waiting_Time_Between_Attempts"
    MAX_ATTEMPTS = "Max_Attempts"

    def __init__(self):
        cfg = yaml.load(open(self.CONFIG_PATH), Loader=yaml.Loader)
        self.log = open(self.LOG_PATH, 'a+')
        self.wait_before_stream_reinitializing = cfg[self.WAITING_TIME_BETWEEN_ATTEMPTS]
        self.max_attempts = cfg[self.MAX_ATTEMPTS]
        self.stream = self.__initialize_stream()
        self.detection_model = mock.MagicMock()
        self.show_in_window = False

    def run(self):
        """
        Main function for fetching stream
        :return: None
        """
        if self.stream:
            while True:
                try:
                    ret, frame = self.stream.read()
                    if ret is True:
                        # TODO: replace by a real function that send frame to detection model
                        self.detection_model.send_image(image=frame)
                        if self.show_in_window:
                            cv2.imshow('frame', frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                except KeyboardInterrupt:
                    self.stream.release()
                    cv2.destroyAllWindows()
                    self.log.close()
                    return None
                except Exception as e:
                    self.stream.release()
                    cv2.destroyAllWindows()
                    self.log.write('Error:Unexpected Error happened:\n {}'.format(e))
                    self.log.close()
                    return None
        else:
            self.log.write("Error initializing stream....\n")
            self.log.close()
            return None

    def __initialize_stream(self):
        """
        Initializing stream from source
        :return: stream or None if couldn't initialize stream
        """
        stream = None
        tries = 0
        try:
            stream = cv2.VideoCapture(self.CHANNEL)
        except:
            self.log.write("Warning:OpenCV Streaming failed - attempt number {}. reinitializing.\n".format(tries))
            tries += 1
            time.sleep(self.wait_before_stream_reinitializing)
            if tries == self.max_attempts:
                self.log.write("Error:Couldn't catch stream.\n")
                return stream
        return stream


if __name__ == '__main__':
    service = WebcamFetcherService()
    service.run()
