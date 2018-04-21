from preprocess import preprocess

if __name__ == "__main__":
    samples, users, movies = preprocess()
    user = users['ur8625456']
    for user in user.values():
