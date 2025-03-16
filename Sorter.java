import java.lang.Math;
import java.util.Random;
public class Sorter {
    int[] a;
    int n;
    private static final Random rand = new Random();

    public Sorter(int[] a){
        this.a = a;
        this.n = a.length;
    }

    public void set(int[] b) {
        a = b;
        n = b.length;
    }

    public int[] copy() {
        int[] b = new int[n];
        for (int i  = 0 ; i < n ; ++ i) {
            b[i] = a[i];
        }
        return b;
    }

    public int[] Bubble() {
        int[] b = this.copy();
        for (int i = 0 ; i < n; ++i) {
            boolean swap = false;
            for (int j = i; j < n - 1; ++j){
                if (b[j] > b[j+1]){
                    int temp = b[j];
                    b[j] = b[j+1];
                    b[j+1] = temp;
                    swap = true;
                }
            }
            if (!swap){
                break;
            }
        }
        return b;
    }

    public int[] Selection() {
        int[] b = this.copy();
        for (int i = 0 ; i < n ; ++i){
            int min = b[i];
            int pos = i;
            for (int j = i + 1; j < n; ++j){
                if (b[j] < min) {
                    pos = j;
                    min = b[j];
                }
            }
            int temp = b[i];
            b[i] = b[pos];
            b[pos] = temp;
        }
        return b;
    }

    public int[] Insertion() {
        int[] b = this.copy();
        for (int i = 0; i < n; ++i) {
            for (int j = i; j > 0; --j) {
                if (b[j] < b[j-1]) {
                    int temp = b[j];
                    b[j] = b[j-1];
                    b[j-1] = temp;
                }
            }
        }
        return b;
    }

    public int[] Merge() {
        int[] b = this.copy();
        return MergeSort(b, 0, n-1);
    }

    public int[] MergeSort(int[] b, int l, int r) {
        if (l == r) {
            return new int[] {b[l]};
        }
        int mid = l + (r - l) / 2;
        int[] left = MergeSort(b, l, mid);
        int[] right = MergeSort(b, mid+1, r);
        int[] temp = new int[r-l+1];
        int i = 0;
        int j = 0;
        int pos = 0;
        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                temp[pos] = left[i];
                i++;
            }
            else{
                temp[pos] = right[j];
                j++;
            }
            pos++;
        }

        while (i < left.length) {
            temp[pos] = left[i];
            i++;
            pos++;
        }

        while (j < right.length) {
            temp[pos]  = right[j];
            j++;
            pos++;
        }

        return temp;

    }

    public int[] Quick() {
        int[] b = this.copy();
        quickSortHelper(b, 0, n - 1);
        return b;
    }

    private void quickSortHelper(int[] arr, int left, int right) {
        if (left < right) {
            int pivotIndex = partition(arr, left, right);
            quickSortHelper(arr, left, pivotIndex - 1);
            quickSortHelper(arr, pivotIndex + 1, right);
        }
    }

    private int partition(int[] arr, int left, int right) {
        int randomIndex = left + rand.nextInt(right - left + 1);
        swap(arr, randomIndex, right);
        int pivot = arr[right];
        int i = left - 1;

        for (int j = left; j < right; j++) {
            if (arr[j] <= pivot) {
                i++;
                swap(arr, i, j);
            }
        }

        swap(arr, i + 1, right);
        return i + 1;
    }

    private void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }



}
