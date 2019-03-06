/*
 * Copyright (c) 2019, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <stdio.h>
#include <stdint.h>

#include "av1/common/blockd.h"
#include "av1/common/onyxc_int.h"

#include "/usr/include/python3.6m/Python.h"

/*
        author：cgy
        date：2018/9/22
        param：
                ppp： 指向
   encoder.c文件中cm->frame_to_show->y_buffer的地址，即重建图像的首地址；
                height：序列的高；
                width：序列的宽；

*/

int init_python() {
  if (!Py_IsInitialized()) {
    const char *mypath = AOM_ROOT
        "/av1/common:"
        "/usr/lib:"
        "/usr/lib/python3.6:"
        "/usr/lib/python3.6/site-packages:"
        "/usr/lib/python3.6/lib-dynload";
    wchar_t *wcsmypath = (wchar_t *)malloc(sizeof(wchar_t) * 1024);
    mbstowcs(wcsmypath, mypath, strlen(mypath));
    Py_SetPath(wcsmypath);
    free(wcsmypath);
    /*
    Py_SetPath(
        AOM_ROOT
        L"/av1/common:"
        "/usr/lib:"
        "/usr/lib/python3.6:"
        "/usr/lib/python3.6/site-packages:"
        "/usr/lib/python3.6/lib-dynload");
        */

    Py_Initialize();

    if (!Py_IsInitialized()) {
      printf("Python init failed!\n");
      return -1;
    }
  }
  return 0;
}

int finish_python() {
  // if (Py_IsInitialized()) {
  //   return Py_FinalizeEx();
  // }
  return 0;
}

uint8_t **call_tensorflow(uint8_t *ppp, int height, int width, int stride,
                          FRAME_TYPE frame_type) {
  PyObject *pModule = NULL;
  PyObject *pFuncI = NULL;
  PyObject *pFuncB = NULL;
  PyObject *pArgs = NULL;

  if (!Py_IsInitialized()) {
    printf("Python init failed!\n");
    return NULL;
  }
  // char *path = NULL;
  // path = getcwd(NULL, 0);
  // printf("current working directory : %s\n", path);
  // free(path);

  // import python
  pModule = PyImport_ImportModule("TEST");
  // pModule = PyImport_ImportModule("TEST_qp52_I");

  // PyEval_InitThreads();
  if (!pModule) {
    printf("don't load Pmodule\n");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire python !\n");
  // 获得TensorFlow函数指针
  pFuncI = PyObject_GetAttrString(pModule, "entranceI");
  if (!pFuncI) {
    printf("don't get I function!");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire entranceFunc !\n");
  pFuncB = PyObject_GetAttrString(pModule, "entranceB");
  if (!pFuncB) {
    printf("don't get B function!");
    Py_Finalize();
    return NULL;
  }

  PyObject *list = PyList_New(height);
  pArgs = PyTuple_New(1);  //以元组方式传参
  PyObject **lists = new PyObject *[height];
  // stringstream ss;
  //将图像缓冲区的数据读到列表中
  for (int i = 0; i < height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < width; j++) {
      PyList_Append(lists[i],
                    Py_BuildValue("i", *(ppp + j)));  //转化为python对象
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 0, list);  //"list" is the input image

  PyObject *presult = NULL;

  // printf("\nstart tensorflow!\n");
  if (frame_type == KEY_FRAME) {
    presult = PyEval_CallObject(pFuncI, pArgs);  //将pArgs参数传递到Python中
  } else {
    presult = PyEval_CallObject(pFuncB, pArgs);  //将pArgs参数传递到Python中
  }

  /*
      Py_ssize_t q = PyList_Size(presult);
      printf("%d", q);
      */

  //需要定义一个二维数组；
  uint8_t **rePic = new uint8_t *[height];
  for (int i = 0; i < height; i++) {
    rePic[i] = new uint8_t[width];
  }
  uint8_t s;

  // FILE *fp = fopen("CPython.yuv", "wb");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "B", &s);
      rePic[i][j] = s;
      // unsigned char uc = (unsigned char)s;
      // fwrite(&uc, 1, 1, fp);
    }
  }
  // fclose(fp);
  return rePic;
}

uint16_t **call_tensorflow_hbd(uint16_t *ppp, int height, int width, int stride,
                               FRAME_TYPE frame_type) {
  PyObject *pModule = NULL;
  PyObject *pFuncI = NULL;
  PyObject *pFuncB = NULL;
  PyObject *pArgs = NULL;

  if (!Py_IsInitialized()) {
    printf("Python init failed!\n");
    return NULL;
  }
  // char *path = NULL;
  // path = getcwd(NULL, 0);
  // printf("current working directory : %s\n", path);
  // free(path);

  pModule = PyImport_ImportModule("TEST");
  // pModule = PyImport_ImportModule("TEST_qp52_I");

  if (!pModule) {
    printf("don't load Pmodule\n");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire python !\n");
  // 获得TensorFlow函数指针
  pFuncI = PyObject_GetAttrString(pModule, "entranceI");
  if (!pFuncI) {
    printf("don't get I function!");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire entranceFunc !\n");
  pFuncB = PyObject_GetAttrString(pModule, "entranceB");
  if (!pFuncB) {
    printf("don't get B function!");
    Py_Finalize();
    return NULL;
  }

  PyObject *list = PyList_New(height);
  pArgs = PyTuple_New(1);  //以元组方式传参
  PyObject **lists = new PyObject *[height];
  // stringstream ss;
  // Read the data from y buffer into the list
  for (int i = 0; i < height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < width; j++) {
      PyList_Append(
          lists[i],
          Py_BuildValue("i", *(ppp + j)));  // Convert to Python objects
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 0, list);  //将列表赋给参数

  PyObject *presult = NULL;

  // printf("\nstart tensorflow!\n");
  if (frame_type == KEY_FRAME) {
    presult = PyEval_CallObject(pFuncI, pArgs);  //将pArgs参数传递到Python中
  } else {
    presult = PyEval_CallObject(pFuncB, pArgs);  //将pArgs参数传递到Python中
  }

  //需要定义一个二维数组；
  uint16_t **rePic = new uint16_t *[height];
  for (int i = 0; i < height; i++) {
    rePic[i] = new uint16_t[width];
  }
  uint16_t s;

  // FILE *fp = fopen("CPython.yuv", "wb");
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      // PyList_GetItem(PyList_GetItem(presult, i), j)意味着presult的(i,j)位置
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "H", &s);
      rePic[i][j] = s;
      // unsigned char uc = (unsigned char)s;
      // fwrite(&uc, 1, 1, fp);
    }
  }
  // fclose(fp);

  // Py_Finalize();//关闭python解释器
  return rePic;
}

void block_call_tensorflow(uint8_t **buf, uint8_t *ppp, int cur_buf_height,
                           int cur_buf_width, int stride,
                           FRAME_TYPE frame_type) {
  PyObject *pModule = NULL;
  PyObject *pFuncI = NULL;
  PyObject *pFuncB = NULL;
  PyObject *pArgs = NULL;

  if (!Py_IsInitialized()) {
    printf("Python init failed!\n");
    return;
  }
  pModule = PyImport_ImportModule("TEST");

  // PyEval_InitThreads();
  if (!pModule) {
    printf("don't load Pmodule\n");
    Py_Finalize();
    return;
  }

  pFuncI = PyObject_GetAttrString(pModule, "entranceI");
  if (!pFuncI) {
    printf("don't get I function!");
    Py_Finalize();
    return;
  }

  pFuncB = PyObject_GetAttrString(pModule, "entranceB");
  if (!pFuncB) {
    printf("don't get B function!");
    Py_Finalize();
    return;
  }
  PyObject *list = PyList_New(cur_buf_height);
  pArgs = PyTuple_New(2);

  PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(AOM_ROOT));

  PyObject **lists = new PyObject *[cur_buf_height];

  for (int i = 0; i < cur_buf_height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < cur_buf_width; j++) {
      PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 1, list);
  PyObject *presult = NULL;
  if (frame_type == KEY_FRAME) {
    presult = PyEval_CallObject(pFuncI, pArgs);
  } else {
    presult = PyEval_CallObject(pFuncB, pArgs);
  }

  for (int i = 0; i < cur_buf_height; i++) {
    for (int j = 0; j < cur_buf_width; j++) {
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "B",
                  &buf[i][j]);
    }
  }
  // Py_Finalize();//关闭python解释器
}

uint16_t **block_call_tensorflow_hbd(uint16_t *ppp, int cur_buf_height,
                                     int cur_buf_width, int stride,
                                     FRAME_TYPE frame_type) {
  PyObject *pModule = NULL;
  PyObject *pFuncI = NULL;
  PyObject *pFuncB = NULL;
  PyObject *pArgs = NULL;

  if (!Py_IsInitialized()) {
    printf("Python init failed!\n");
    return NULL;
  }

  // char *path = NULL;
  // path = getcwd(NULL, 0);
  // printf("current working directory : %s\n", path);
  // free(path);

  pModule = PyImport_ImportModule("TEST");
  // pModule = PyImport_ImportModule("TEST_qp52_B");
  // pModule = PyImport_ImportModule("TEST_qp52_I");

  // PyEval_InitThreads();
  if (!pModule) {
    printf("don't load Pmodule\n");
    Py_Finalize();
    return NULL;
  }

  pFuncI = PyObject_GetAttrString(pModule, "entranceI");
  if (!pFuncI) {
    printf("don't get I function!");
    Py_Finalize();
    return NULL;
  }
  // printf("succeed acquire entranceFunc !\n");
  pFuncB = PyObject_GetAttrString(pModule, "entranceB");
  if (!pFuncB) {
    printf("don't get B function!");
    Py_Finalize();
    return NULL;
  }
  PyObject *list = PyList_New(cur_buf_height);
  pArgs = PyTuple_New(1);
  PyObject **lists = new PyObject *[cur_buf_height];

  for (int i = 0; i < cur_buf_height; i++) {
    lists[i] = PyList_New(0);
    for (int j = 0; j < cur_buf_width; j++) {
      PyList_Append(lists[i], Py_BuildValue("i", *(ppp + j)));
    }
    PyList_SetItem(list, i, lists[i]);
    ppp += stride;
    // PyList_Append(list, lists[i]);
  }
  PyTuple_SetItem(pArgs, 0, list);
  PyObject *presult = NULL;
  if (frame_type == KEY_FRAME) {
    presult = PyEval_CallObject(pFuncI, pArgs);
  } else {
    presult = PyEval_CallObject(pFuncB, pArgs);
  }

  uint16_t **rePic = new uint16_t *[cur_buf_height];
  for (int i = 0; i < cur_buf_height; i++) {
    rePic[i] = new uint16_t[cur_buf_width];
  }
  uint16_t s;

  // FILE *fp = fopen("CPython.yuv", "wb");
  for (int i = 0; i < cur_buf_height; i++) {
    for (int j = 0; j < cur_buf_width; j++) {
      // PyList_GetItem(PyList_GetItem(presult, i), j) mean presult(i,j)
      PyArg_Parse(PyList_GetItem(PyList_GetItem(presult, i), j), "H", &s);
      rePic[i][j] = s;
      // unsigned char uc = (unsigned char)s;
      // fwrite(&uc, 1, 1, fp);
    }
  }
  // fclose(fp);

  // Py_Finalize();
  return rePic;
}
