#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

long long comparisons = 0; // global counter

// Insertion sort for small arrays
void insertion_sort(int arr[], int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= left && arr[j] > key) {
            comparisons++;
            arr[j + 1] = arr[j];
            j--;
        }
        if (j >= left) comparisons++; // final failed comparison
        arr[j + 1] = key;
    }
}

// Merge function
void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;

    while (i < n1 && j < n2) {
        comparisons++;
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

// Hybrid merge sort
void hybrid_merge_sort(int arr[], int l, int r, int S) {
    if (r - l + 1 <= S) {
        insertion_sort(arr, l, r);
    } else if (l < r) {
        int m = (l + r) / 2;
        hybrid_merge_sort(arr, l, m, S);
        hybrid_merge_sort(arr, m + 1, r, S);
        merge(arr, l, m, r);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 128) {
        if (rank == 0) {
            fprintf(stderr, "Error: need exactly 128 ranks (got %d)\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    // Input sizes to test
    long test_sizes[] = {1000, 2000, 5000, 10000, 100000, 1000000, 10000000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    // Each rank gets a unique S
    int S = rank + 1;

    // Print CSV header
    if (rank == 0) {
        printf("n,S,Comparisons,Time(s)\n");
    }

    for (int t = 0; t < num_tests; t++) {
        long n = test_sizes[t];

        // Allocate array on all ranks
        int *arr = malloc(n * sizeof(int));
        if (!arr) {
            fprintf(stderr, "Rank %d: malloc failed for n=%ld\n", rank, n);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Rank 0 generates input
        if (rank == 0) {
            srand(42 + t); // fixed but different seed per test size
            for (long i = 0; i < n; i++) {
                arr[i] = rand() % n;
            }
        }

        // Broadcast array to all ranks
        MPI_Bcast(arr, n, MPI_INT, 0, MPI_COMM_WORLD);

        // Reset comparisons
        comparisons = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        hybrid_merge_sort(arr, 0, n - 1, S);
        double end = MPI_Wtime();

        double elapsed = end - start;

        // Gather results to rank 0
        long long *all_comparisons = NULL;
        double *all_times = NULL;
        if (rank == 0) {
            all_comparisons = malloc(size * sizeof(long long));
            all_times = malloc(size * sizeof(double));
        }

        MPI_Gather(&comparisons, 1, MPI_LONG_LONG,
                   all_comparisons, 1, MPI_LONG_LONG,
                   0, MPI_COMM_WORLD);
        MPI_Gather(&elapsed, 1, MPI_DOUBLE,
                   all_times, 1, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        // Print results
        if (rank == 0) {
            for (int i = 0; i < size; i++) {
                printf("%ld,%d,%lld,%.6f\n", n, i + 1, all_comparisons[i], all_times[i]);
            }
            free(all_comparisons);
            free(all_times);
        }

        free(arr);
    }

    MPI_Finalize();
    return 0;
}
