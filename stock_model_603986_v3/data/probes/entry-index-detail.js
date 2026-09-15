/*
* @description: require.js 指数系列入口
* @author: yuanfan
* @update: yuanfan (2019-07-30)
*/

//Load common code that includes config, then load the app logic for this page.
requirejs(['../common'], function (common) {
    requirejs(['../app/index-detail']);
});