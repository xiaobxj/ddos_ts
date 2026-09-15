/*
 * @description: require.js 公用配置文件
 * @author: yuanfan
 * @update: yuanfan (2019-7-30)
 */
window.hqUrl = 'https://hq.cnindex.com.cn'; // 生产环境接口域名
window.domainUrl = 'cnindex.com.cn';

//是否为 TRS 页面
window.isTRS = true;
var url = window.location.href,
  transArr = url.split('/');
if (!getCookie('language')) {
  if (transArr.indexOf('eng') !== -1 || transArr.indexOf('en') !== -1) {
    setCookie('language', 'en');
  } else {
    setCookie('language', 'zh_CN');
  }
} else {
  if (window.isTRS) {
    if (transArr.indexOf('eng') !== -1 || transArr.indexOf('en') !== -1) {
      setCookie('language', 'en');
    } else {
      setCookie('language', 'zh_CN');
    }
  } else {
    if (transArr.indexOf('en') !== -1) {
      setCookie('language', 'en');
    } else {
      setCookie('language', 'zh_CN');
    }
  }
}
//全局变量，是否为英文
window.isEN = getCookie('language') === 'en';
//全局变量，是否为移动端
window.isMobile = isMobile();

require.config({
  //baseUrl: '../../js/lib',
  packages: [
    {
      name: 'highcharts',
      main: 'highcharts'
    }
  ],
  paths: {
    jquery: '../lib/jquery',
    vue: '../lib/vue.min',
    ELEMENT: '../lib/element-ui/index',
    'element-en': '../lib/element-ui/en',
    text: '../lib/require-text',
    storageTiming: '../lib/storage-timing.min',
    header: window.isEN === true ? '/en/common/header.html' : '/common/header.html',
    mheader: window.isEN === true ? '/en/common/m-header.html' : '/common/m-header.html',
    nav: window.isEN === true ? '/en/common/nav.html' : '/common/nav.html',
    mnav: window.isEN === true ? '/en/common/m-nav.html' : '/common/m-nav.html',
    footer: window.isEN === true ? '/en/common/footer.html' : '/common/footer.html',
    backtop: window.isEN === true ? '/en/common/backtop.html' : '/common/backtop.html',
    download: window.isEN === true ? '/en/common/market-download.html' : '/common/market-download.html',
    echarts: '../lib/echarts.min',
    highcharts: '../lib/highstock',
    //图表组件封装
    stockChart: '../app/stock-chart-plugin',
    echartsPlugin: '../app/echarts-plugin',
    //公用组件
    NsdkRequest: '/js/nsdk-request',
    ultls: '/common/ultls',
    area: '/common/area'
  },
  shim: {},
  waitSeconds: 0
});

/**
 * 设置cookie
 * @param name cookie的名称
 * @param value cookie的值
 * @param day cookie的过期时间
 */
function setCookie(name, value, day) {
  if (day !== 0) {
    //当设置的时间等于0时，不设置expires属性，cookie在浏览器关闭后删除
    var expires = day * 24 * 60 * 60 * 1000;
    var date = new Date(+new Date() + expires);
    document.cookie = name + '=' + escape(value) + '; path=/' + ';expires=' + date.toUTCString() + ';domain=' + window.domainUrl;
  } else {
    document.cookie = name + '=' + escape(value) + ';domain=' + window.domainUrl;
  }
}
/**
 * 获取对应名称的cookie
 * @param name cookie的名称
 * @returns {null} 不存在时，返回null
 */
function getCookie(name) {
  var arr;
  var reg = new RegExp('(^| )' + name + '=([^;]*)(;|$)');
  if ((arr = document.cookie.match(reg))) return unescape(arr[2]);
  else return null;
}
/*删除cookie*/
function delCookie(name) {
  var exp = new Date();
  exp.setTime(exp.getTime() - 1);
  var cval = getCookie(name);
  if (cval !== null) {
    document.cookie = name + '=' + cval + '; expires=' + exp.toUTCString() + '; path=/;';
  }
}

//是否为移动端
function isMobile() {
  var mobileArry = ['iPhone', 'iPad', 'Android', 'Windows Phone', 'BB10; Touch'];
  var ua = navigator.userAgent;

  var res = mobileArry.filter(function (arr) {
    return ua.indexOf(arr) > 0;
  });

  return res.length > 0;
}
