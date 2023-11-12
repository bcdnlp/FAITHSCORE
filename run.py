from faithscore.framework import FaithScore
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path',
                        type=str,)

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--vem_type',
                        type=str,
                        choices=["ofa", "ofa-ve", "llava"],
                        default="llava")
    parser.add_argument('--llava_path',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--llama_path',
                        type=str,
                        default=".cache/factscore/")

    parser.add_argument('--use_llama',
                        type=bool,
                        default=False)

    args = parser.parse_args()

    images = []
    answers = []

    with open(args.answer_path) as f:
        for line in f:
            line = eval(line)
            answers.append(line["answer"])
            images.append(line["image"])

    score = FaithScore(vem_type=args.vem_type, api_key=args.openai_key,
                       llava_path=args.llava_path, use_llama=args.use_llama,
                       llama_path=args.llama_path)
    f, sentence_f = score.faithscore(answers, images)
    print(f"Faithscore is {f}. Sentence-level faithscore is {sentence_f}.")
