#include <stdio.h>

void main() {
    static int x[] = {1,2,3,4};
    static int y[] = {10, 20, 30, 40};
    int i, z[4];
    for (i=0; i<4; ++i)
    z[i] = x[i] + y[3-i];
    printf("�ݴ� ��ġ�� �迭��� ��\n");
    for(i = 0; i<4; ++i)
    printf("%d + %d = %d\n", x[i], y[3-i], z[i]);

}