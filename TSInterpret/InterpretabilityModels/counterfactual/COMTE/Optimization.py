class BruteForceSearch(BaseExplanation):
    def _find_best(self, x_test, distractor, label_idx):
        global CLASSIFIER
        global X_TEST
        global DISTRACTOR
        CLASSIFIER = self.clf
        X_TEST = x_test
        DISTRACTOR = distractor
        input_ = x_test.reshape(1, -1, self.window_size)
        best_case = self.clf(input_)[0][label_idx]
        best_column = None
        tuples = []
        for c in range(0, self.channels):
            if np.any(distractor[0][c] != x_test[0][c]):
                tuples.append((c, label_idx))
        if self.threads == 1:
            results = []
            for t in tuples:
                results.append(_eval_one(t))
        else:
            pool = multiprocessing.Pool(self.threads)
            results = pool.map(_eval_one, tuples)
            pool.close()
            pool.join()
        for (c, _), pred in zip(tuples, results):
            if pred > best_case:
                best_column = c
                best_case = pred
        if not self.silent:
            logging.info("Best column: %s, best case: %s", best_column, best_case)
        return best_column, best_case

    def explain(self, x_test, to_maximize=None, num_features=10):
        input_ = x_test.reshape(1, -1, self.window_size)
        orig_preds = self.clf(input_)
        if to_maximize is None:
            to_maximize = np.argsort(orig_preds)[0][-2:-1][0]

        orig_label = np.argmax(self.clf(input_))
        if orig_label == to_maximize:
            print("Original and Target Label are identical !")
            return None, None
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors
        )
        # print('distractores',distractors)
        best_explanation = set()
        best_explanation_score = 0
        for count, dist in enumerate(distractors):
            explanation = []
            modified = x_test.copy()
            prev_best = 0
            # best_dist = dist
            while True:
                input_ = modified.reshape(1, -1, self.window_size)
                probas = self.clf(input_)
                # print('Current may',np.argmax(probas))
                # print(to_maximize)
                if np.argmax(probas) == to_maximize:
                    current_best = np.max(probas)
                    if current_best > best_explanation_score:
                        best_explanation = explanation
                        best_explanation_score = current_best
                    if current_best <= prev_best:
                        break
                    prev_best = current_best
                    if not self.dont_stop:
                        break
                if (
                    not self.dont_stop
                    and len(best_explanation) != 0
                    and len(explanation) >= len(best_explanation)
                ):
                    break
                best_column, _ = self._find_best(modified, dist, to_maximize)
                if best_column is None:
                    break

                modified[0][best_column] = dist[0][best_column]
                explanation.append(best_column)

        other = modified
        target = np.argmax(self.clf(other), axis=1)
        return other, target





class OptimizedSearch(BaseExplanation):
    def __init__(
        self,
        clf,
        timeseries,
        labels,
        silent,
        threads,
        num_distractors,
        max_attempts,
        maxiter,
        **kwargs,
    ):
        super().__init__(clf, timeseries, labels, **kwargs)
        self.discrete_state = False
        self.backup = BruteForceSearch(clf, timeseries, labels, **kwargs)
        self.max_attemps = max_attempts
        self.maxiter = maxiter

    def opt_Discrete(self, to_maximize, x_test, dist, columns, init, num_features=None):
        fitness_fn = LossDiscreteState(
            to_maximize,
            self.clf,
            x_test,
            dist,
            columns,
            reg=0.8,
            max_features=num_features,
            maximize=False,
        )
        problem = mlrose.DiscreteOpt(
            length=len(columns), fitness_fn=fitness_fn, maximize=False, max_val=2
        )
        best_state, best_fitness = mlrose.random_hill_climb(
            problem,
            max_attempts=self.max_attemps,
            max_iters=self.maxiter,
            init_state=init,
            restarts=5,
        )

        self.discrete_state = True
        return best_state

    def _prune_explanation(
        self, explanation, x_test, dist, to_maximize, max_features=None
    ):
        if max_features is None:
            max_features = len(explanation)
        short_explanation = set()
        while len(short_explanation) < max_features:
            modified = x_test.copy()
            for c in short_explanation:
                modified[0][c] = dist[0][c]
            input_ = modified.reshape(1, -1, self.window_size)
            prev_proba = self.clf(input_)[0][to_maximize]
            best_col = None
            best_diff = 0
            for c in explanation:
                tmp = modified.copy()

                tmp[0][c] = dist[0][c]

                input_ = tmp.reshape(1, self.channels, self.window_size)
                cur_proba = self.clf(input_)[0][to_maximize]
                if cur_proba - prev_proba > best_diff:
                    best_col = c
                    best_diff = cur_proba - prev_proba
            if best_col is None:
                break
            else:
                short_explanation.add(best_col)
        return short_explanation

    def explain(
        self, x_test, num_features=None, to_maximize=None
    ) -> Tuple[np.array, int]:
        input_ = x_test.reshape(1, -1, self.window_size)
        orig_preds = self.clf(input_)

        orig_label = np.argmax(orig_preds)

        if to_maximize is None:
            to_maximize = np.argsort(orig_preds)[0][-2:-1][0]

        # print('Current may',np.argmax(orig_preds))
        # print(to_maximize)

        if orig_label == to_maximize:
            print("Original and Target Label are identical !")
            return None, None

        explanation = self._get_explanation(x_test, to_maximize, num_features)
        tr, _ = explanation
        if tr is None:
            print("Run Brute Force as Backup.")
            explanation = self.backup.explain(
                x_test, num_features=num_features, to_maximize=to_maximize
            )
        best, other = explanation
        # print('Other',np.array(other).shape)
        # print('Best',np.array(best).shape)
        target = np.argmax(self.clf(best), axis=1)

        return best, target

    def _get_explanation(self, x_test, to_maximize, num_features):
        distractors = self._get_distractors(
            x_test, to_maximize, n_distractors=self.num_distractors
        )

        # Avoid constructing KDtrees twice
        self.backup.per_class_trees = self.per_class_trees
        self.backup.per_class_node_indices = self.per_class_node_indices

        best_explanation = set()
        best_explanation_score = 0

        for count, dist in enumerate(distractors):
            columns = [
                c for c in range(0, self.channels) if np.any(dist[0][c] != x_test[0][c])
            ]

            # Init options
            init = [0] * len(columns)

            result = self.opt_Discrete(
                to_maximize, x_test, dist, columns, init=init, num_features=num_features
            )

            if not self.discrete_state:
                explanation = {
                    x for idx, x in enumerate(columns) if idx in np.nonzero(result.x)[0]
                }
            else:
                explanation = {
                    x for idx, x in enumerate(columns) if idx in np.nonzero(result)[0]
                }

            explanation = self._prune_explanation(
                explanation, x_test, dist, to_maximize, max_features=num_features
            )

            modified = x_test.copy()

            for c in columns:
                if c in explanation:
                    modified[0][c] = dist[0][c]
            input_ = modified.reshape(1, -1, self.window_size)
            probas = self.clf(input_)

            if not self.silent:
                logging.info("Current probas: %s", probas)

            if np.argmax(probas) == to_maximize:
                current_best = np.max(probas)
                if current_best > best_explanation_score:
                    best_explanation = explanation
                    best_explanation_score = current_best
                    best_modified = modified

        if len(best_explanation) == 0:
            return None, None

        return best_modified, best_explanation
