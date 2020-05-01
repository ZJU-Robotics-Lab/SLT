import numpy as np

"""
Calculate Word Error Rate
Word Error Rate = (Substitutions + Insertions + Deletions) / Number of Words Spoken
Reference:
https://holianh.github.io/portfolio/Cach-tinh-WER/
https://github.com/imalic3/python-word-error-rate
"""
def wer(r, h):
    # initialisation
    d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return float(d[len(r)][len(h)]) / len(r) * 100


if __name__ == '__main__':
    # Calculate WER
    r = [1,2,3,4]
    h = [1,1,3,5,6]
    print(wer(r, h))
