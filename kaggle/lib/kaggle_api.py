from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleAPI(KaggleApi):
    def __init__(self, api_client=None):
        super(KaggleAPI, self).__init__(api_client=api_client)
        self.api = KaggleApi()
        self.api.authenticate()

    def submit(self, competition, file_name, message):
        return self.api.competition_submit(
            competition=competition, file=file_name, message=message)

    def get_last_score(self, competition):
        items = self.api.competition_submissions(competition=competition)
        return items[0].publicScore if items else None

    def get_best_score(self, competition):
        items = self.api.competition_submissions(competition=competition)
        best_score = None
        for item in items:
            if not best_score or item.publicScore and float(
                    best_score) < float(str(item.publicScore)):
                best_score = str(item.publicScore)
        return best_score


if __name__ == '__main__':
    api = KaggleAPI()
    score = api.get_best_score(competition='digit-recognizer')
    print(score)

    score = api.get_last_score(competition='digit-recognizer')
    print(score)
