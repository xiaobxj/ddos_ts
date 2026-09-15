/**
 * Created by yuanfan on 2019/7/30.
 */

requirejs(
  [
    'highcharts/highstock',
    'jquery',
    'vue',
    'ELEMENT',
    'stockChart',
    'echartsPlugin',
    'highcharts/highcharts-zh_CN',
    'highcharts/theme',
    'highcharts/modules/exporting',
    'highcharts/indicators/indicators-all',
    'highcharts/modules/map',
    'highcharts/modules/no-data-to-display',
    'ultls',
    'NsdkRequest'
  ],
  function (
    Highcharts,
    $,
    Vue,
    ELEMENT,
    stockChart,
    echartsPlugin,
    ZhCNData,
    themeData,
    exporting,
    indicators,
    maps,
    noDataToDisplay,
    ultls,
    NsdkRequest
  ) {
    window.mobileType = true;
    require(['text!header', 'text!mheader', 'text!nav', 'text!mnav', 'text!footer', 'text!backtop', 'text!download'], function (
      headerDom,
      mHeaderDom,
      navDom,
      mNavDom,
      footerDom,
      backtopDom,
      downloadDom
    ) {
      require([
        window.isEN === true ? '../../en/common/nav' : '../../common/nav',
        '../../common/header',
        window.isEN === true ? '../../en/common/footer' : '../../common/footer',
        '../../common/backtop',
        '../../common/market-download'
      ]);
      if (window.isMobile) {
        $('.container').prepend(mHeaderDom);
        window.isEN === true ? $('#en-header').after(mNavDom) : $('#header').after(mNavDom);
      } else {
        $('.container').prepend(headerDom);
        window.isEN === true ? $('#en-header').after(navDom) : $('#header').after(navDom);
      }
      $('.container').append(footerDom);
      $('body').append(backtopDom);
    });
    var colors = ['#0181f2', '#a6e0fd', '#bc95f4', '#faa9ae', '#fbce9d'];
    //indexCode 初始化
    var indexCode = ultls.getQueryString('indexCode');
    //特殊指数
    var exCodeList = [
      '399415',
      '399416',
      '399354',
      '399297',
      'CN5075',
      'CN6075',
      '970027',
      '470027',
      '970030',
      '470030',
      '970033',
      '470033',
      '399378',
      'CN2378',
      '970028',
      '470028',
      '970031',
      '470031',
      '970034',
      '470034',
      '970026',
      '470026',
      '970029',
      '470029',
      '970032',
      '470032'
    ];

    // 2个国证恒生指数代码
    const hsiCodeList = ['SZHKDE', 'SZHKCON'];

    var sampleShow = exCodeList.indexOf(indexCode.toUpperCase()) > -1;

    if (window.isEN) {
      ELEMENT.locale({
        el: {
          colorpicker: {
            confirm: 'OK',
            clear: 'Clear'
          },
          datepicker: {
            now: 'Now',
            today: 'Today',
            cancel: 'Cancel',
            clear: 'Clear',
            confirm: 'OK',
            selectDate: 'Select date',
            selectTime: 'Select time',
            startDate: 'Start Date',
            startTime: 'Start Time',
            endDate: 'End Date',
            endTime: 'End Time',
            prevYear: 'Previous Year',
            nextYear: 'Next Year',
            prevMonth: 'Previous Month',
            nextMonth: 'Next Month',
            year: '',
            month1: 'January',
            month2: 'February',
            month3: 'March',
            month4: 'April',
            month5: 'May',
            month6: 'June',
            month7: 'July',
            month8: 'August',
            month9: 'September',
            month10: 'October',
            month11: 'November',
            month12: 'December',
            week: 'week',
            weeks: {
              sun: 'Sun',
              mon: 'Mon',
              tue: 'Tue',
              wed: 'Wed',
              thu: 'Thu',
              fri: 'Fri',
              sat: 'Sat'
            },
            months: {
              jan: 'Jan',
              feb: 'Feb',
              mar: 'Mar',
              apr: 'Apr',
              may: 'May',
              jun: 'Jun',
              jul: 'Jul',
              aug: 'Aug',
              sep: 'Sep',
              oct: 'Oct',
              nov: 'Nov',
              dec: 'Dec'
            }
          },
          select: {
            loading: 'Loading',
            noMatch: 'No matching data',
            noData: 'No data',
            placeholder: 'Select'
          },
          cascader: {
            noMatch: 'No matching data',
            loading: 'Loading',
            placeholder: 'Select',
            noData: 'No data'
          },
          pagination: {
            goto: 'Go to',
            pagesize: '/page',
            total: 'Total {total}',
            pageClassifier: ''
          },
          messagebox: {
            title: 'Message',
            confirm: 'OK',
            cancel: 'Cancel',
            error: 'Illegal input'
          },
          upload: {
            deleteTip: 'press delete to remove',
            delete: 'Delete',
            preview: 'Preview',
            continue: 'Continue'
          },
          table: {
            emptyText: 'No Data',
            confirmFilter: 'Confirm',
            resetFilter: 'Reset',
            clearFilter: 'All',
            sumText: 'Sum'
          },
          tree: {
            emptyText: 'No Data'
          },
          transfer: {
            noMatch: 'No matching data',
            noData: 'No data',
            titles: ['List 1', 'List 2'], // to be translated
            filterPlaceholder: 'Enter keyword', // to be translated
            noCheckedFormat: '{total} items', // to be translated
            hasCheckedFormat: '{checked}/{total} checked' // to be translated
          },
          image: {
            error: 'FAILED'
          },
          pageHeader: {
            title: 'Back' // to be translated
          }
        }
      });
    }

    //Vue 初始化
    Vue.use(ELEMENT);
    //时间格式过滤
    Vue.filter('dateFormat', function (value) {
      return ultls.formateDate(value);
    });
    //初始化图表数据
    var timeChart = {},
      timeChart2 = {};

    // 昨日收盘价
    var preClosePrice = '';

    //初始化 timeData (一天分时数据)
    //       ohcl （历史行情 K 线所需数据）
    //       newHistoryData (历史行情折线数据)
    //       newHistoryData (历史行情折线数据)
    var timeData = [],
      ohclData = [],
      newHistoryData = [],
      volumeData = [];
    //地图组件（用于缩放）
    maps(Highcharts);
    //无数据时
    noDataToDisplay(Highcharts);
    //导出模块
    exporting(Highcharts);
    //技术指标模块
    indicators(Highcharts);
    //设置中文
    Highcharts.setOptions(window.isEN ? ZhCNData.ENData : ZhCNData.ZhCNData);
    //皮肤颜色

    Highcharts.setOptions({
      global: { useUTC: false },
      lang: {
        rangeSelectorZoom: '' // 不显示 'zoom' 文字
      }
    });

    //初始化热门指数推荐
    var hotIndexList = [
      {
        indexCode: '399001',
        indexFullCname: null,
        indexFullEname: null,
        shortCName: '深证成指',
        shortEName: 'Shenzhen Component'
      },
      {
        indexCode: '399330',
        indexFullCname: null,
        indexFullEname: null,
        shortCName: '深证100',
        shortEName: 'Shenzhen 100'
      },
      {
        indexCode: '399006',
        indexFullCname: null,
        indexFullEname: null,
        shortCName: '创业板指',
        shortEName: 'ChiNext Index'
      },
      {
        indexCode: '399317',
        indexFullCname: null,
        indexFullEname: null,
        shortCName: '国证A指',
        shortEName: 'CNI A Share'
      },
      {
        indexCode: '399311',
        indexFullCname: null,
        indexFullEname: null,
        shortCName: '国证1000',
        shortEName: 'CNI 1000'
      },
      {
        indexCode: '000001',
        indexFullCname: null,
        indexFullEname: null,
        shortCName: '上证指数',
        shortEName: 'SSE Index'
      },
      {
        indexCode: '399300',
        indexFullCname: null,
        indexFullEname: null,
        shortCName: '沪深300',
        shortEName: 'CSI 300'
      },
      {
        indexCode: '399905',
        indexFullCname: null,
        indexFullEname: null,
        shortCName: '中证500',
        shortEName: 'CSI 500'
      }
    ];
    //指数指标配置
    var macdOption = {
        type: 'macd',
        id: 'macd',
        yAxis: 2,
        linkedTo: 'main',
        lineWidth: 1,
        color: '#8c8c8c',
        macdLine: {
          styles: {
            lineColor: '#ad6eff'
          },
          zones: {
            fillColor: '#ad6eff'
          }
        },
        signalLine: {
          styles: {
            lineColor: '#ffa33f'
          }
        },
        states: {
          hover: {
            enabled: false
          }
        }
      },
      bollOption = {
        id: 'bb',
        type: 'bb',
        yAxis: 2,
        linkedTo: 'main',
        lineWidth: 1,
        bottomLine: {
          styles: {
            lineColor: '#ad6eff'
          }
        },
        topLine: {
          styles: {
            lineColor: '#ffa33f'
          }
        }
      },
      rocOption = {
        type: 'roc',
        id: 'roc',
        yAxis: 2,
        linkedTo: 'main',
        lineWidth: 1,
        color: '#8c8c8c',
        macdLine: {
          styles: {
            lineColor: '#ad6eff'
          },
          zones: {
            fillColor: '#ad6eff'
          }
        },
        signalLine: {
          styles: {
            lineColor: '#ffa33f'
          }
        },
        states: {
          hover: {
            enabled: false
          }
        }
      },
      rsiOption = {
        type: 'rsi',
        id: 'rsi',
        yAxis: 2,
        linkedTo: 'main',
        lineWidth: 1,
        states: {
          hover: {
            enabled: false
          }
        }
      };

    // 创建分时图 type: line or area
    var createTimeChart = function (timeData, type, gapType, flag) {
      const gaps = { m1: 1, m15: 15, m30: 30, m60: 60 };
      const gap = gaps[gapType] || 1;
      if (!$.isEmptyObject(timeChart)) {
        timeChart.destroy();
        // timeChart2.destroy();
      }
      var option = {
        chart: {
          zoomType: 'x',
          marginTop: window.isMobile ? 0 : 86,
          marginLeft: 0,
          marginRight: 40,
          backgroundColor: 'transparent'
        },
        exporting: {
          url: '/export',
          enabled: !window.isMobile,
          //allowHTML: true,
          buttons: {
            contextButton: {
              enabled: false
            },
            exportButton: {
              contextButton: {
                enabled: false
              },
              symbol: window.isEN ? 'url(/image/en-save-img-btn.jpg)' : 'url(/image/save-img-btn.png)',
              //text: window.isEN ? 'Save': '保存',
              menuItems: ['downloadPNG', 'downloadJPEG', 'downloadSVG'],
              y: 8,
              x: -10,
              width: 100
            }
          },
          filename: indexCode + '_' + ultls.getTodayValue()
        },
        navigation: {
          buttonOptions: {
            symbolSize: 85,
            symbolY: 54,
            symbolX: 110,
            //symbolStrokeWidth: 1,

            theme: {
              // 'stroke-width': 1,
              //  stroke: '#d9d9d9',
              r: 0,
              states: {
                hover: {
                  fill: '#fff'
                },
                select: {
                  fill: '#fff'
                }
              }
            }
          }
        },
        navigator: {
          enabled: false
        },
        lang: {
          rangeSelectorZoom: ''
        },
        credits: {
          enabled: false
        },
        scrollbar: {
          enabled: false
        },
        rangeSelector: {
          // buttonTheme: {
          //     display: 'none'
          // },
          inputEnabled: false,
          enabled: false
        },
        xAxis: {
          alternateGridColor: '#F4F4F4',
          showFirstLabel: true,
          showLastLabel: true,
          labels: {
            formatter: function () {
              var returnTime = Highcharts.dateFormat('%H:%M', this.value);
              return returnTime;
            }
          }
        },
        yAxis: [
          {
            left: '40',
            plotLines: [
              {
                value: flag,
                color: '#999',
                width: 2,
                dashStyle: 'shortdash',
                zIndex: 2040,
                label: {
                  align: 'left',
                  style: {
                    backgroundColor: '#999',
                    color: '#999'
                  },
                  // text: flag ? flag.toFixed(0): '' ,
                  text: flag ? flag.toFixed(2) : '',
                  y: 16,
                  x: 4,

                  color: '#999'
                }
              }
            ],
            labels: {
              formatter: function () {
                return this.value.toFixed(0);
                // return this.value.toFixed(2)
              }
            }
          }
        ],
        title: {
          text: ''
        },
        plotOptions: {
          series: {
            showInLegend: true,
            gapSize: 10000,
            dataGrouping: {
              enabled: false
            }
          }
        },
        tooltip: {
          crosshairs: [true, true],
          shape: 'square',
          split: true,
          shared: true,
          xDateFormat: '%Y-%m-%d %H:%M:%S',
          borderColor: '6495ed',
          useHTML: true,
          formatter: function () {
            var s =
              '<div class="chart-tooltip"><b>' +
              this.points[0].point.series.name +
              '</b><br/>' +
              '<span style="this.points[0].point.series.color">' +
              Highcharts.dateFormat('%Y-%m-%d %H:%M', this.x) +
              '</span><br/>' +
              '点位：' +
              '<span style="this.points[0].point.series.color">' +
              this.y.toFixed(2) +
              '</span><br/>';
            //涨跌趋势渲染
            if (this.points[0].point.extra[6]) {
              if (this.points[0].point.extra[6] > 0) {
                s +=
                  '涨跌：' +
                  '<span>+' +
                  this.points[0].point.extra[6].toFixed(2) +
                  '(+' +
                  (this.points[0].point.extra[7] * 100).toFixed(2) +
                  '%)' +
                  '</span><br/>';
              } else {
                s +=
                  '涨跌：' +
                  '<span>' +
                  this.points[0].point.extra[6].toFixed(2) +
                  '(' +
                  (this.points[0].point.extra[7] * 100).toFixed(2) +
                  '%)' +
                  '</span><br/>';
              }
            } else {
            }
            s +=
              '成交额：' +
              '<span>' +
              (!vm.arrEmptyData.includes(this.points[0].point.extra[8]) ? ultls.moneyUnits(this.points[0].point.extra[8]) + '元' : '--') +
              '</span><br/>';
            //债券指数
            if (vm.newIndexType == true) {
              s +=
                '成交量：' +
                '<span>' +
                (!vm.arrEmptyData.includes(this.points[0].point.extra[9]) ? ultls.volumeBondUnits(this.points[0].point.extra[9]) : '--') +
                '</span><br/>';
            } else {
              //非债券指数
              s +=
                '成交量：' +
                '<span>' +
                (!vm.arrEmptyData.includes(this.points[0].point.extra[9]) ? ultls.volumeSmallUnits(this.points[0].point.extra[9]) : '--') +
                '</span><br/>';
            }

            s += '</div>';

            var sEN =
              '<div class="chart-tooltip"><b>' +
              this.points[0].point.series.name +
              '</b><br/>' +
              '<span style="this.points[0].point.series.color">' +
              Highcharts.dateFormat('%Y-%m-%d %H:%M', this.x) +
              '</span><br/>' +
              'value：' +
              '<span style="this.points[0].point.series.color">' +
              this.y.toFixed(2) +
              '</span><br/>';
            //涨跌趋势渲染
            if (this.points[0].point.extra[6]) {
              if (this.points[0].point.extra[6] > 0) {
                sEN +=
                  'Change：' +
                  '<span>+' +
                  this.points[0].point.extra[6].toFixed(2) +
                  '(+' +
                  (this.points[0].point.extra[7] * 100).toFixed(2) +
                  '%)' +
                  '</span><br/>';
              } else {
                sEN +=
                  'Change：' +
                  '<span>' +
                  this.points[0].point.extra[6].toFixed(2) +
                  '(' +
                  (this.points[0].point.extra[7] * 100).toFixed(2) +
                  '%)' +
                  '</span><br/>';
              }
            }

            sEN +=
              'Turnover：' +
              '<span>' +
              (!vm.arrEmptyData.includes(this.points[0].point.extra[8]) ? ultls.moneyUnits(this.points[0].point.extra[8]) : '--') +
              '</span><br/>';

            //债券指数
            if (vm.newIndexType) {
              sEN +=
                'Volume：' +
                '<span>' +
                (!vm.arrEmptyData.includes(this.points[0].point.extra[9]) ? ultls.volumeBondUnits(this.points[0].point.extra[9]) : '--') +
                '</span><br/>';
            } else {
              //非债券指数
              sEN +=
                'Volume：' +
                '<span>' +
                (!vm.arrEmptyData.includes(this.points[0].point.extra[9]) ? ultls.volumeSmallUnits(this.points[0].point.extra[9]) : '--') +
                '</span><br/>';
            }
            sEN += '</div>';
            return window.isEN ? sEN : s;
          }
        },
        series: [
          {
            type: type,
            data: timeDateFormate(timeData, gap),
            name: vm.indexName,
            threshold: null,
            lineWidth: 2,
            color: colors[0],
            yAxis: 0,
            fillColor: {
              linearGradient: {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 1
              },
              stops: [
                [0, Highcharts.getOptions().colors[0]],
                [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
              ]
            }
          }
        ]
      };

      if (!window.isMobile) {
        timeChart = Highcharts.stockChart('chart1', option);
        timeChart2 = Highcharts.stockChart('chart2', option);
      } else {
        timeChart = Highcharts.stockChart('mchart', option);
      }
    };

    // 创建国证恒生指数的走势图
    var createHsiIndexChart = function (arrChartData, showType) {
      if (!$.isEmptyObject(timeChart)) {
        timeChart.destroy();
      }
      var option = {
        chart: {
          zoomType: 'x',
          marginTop: window.isMobile ? 0 : 86,
          marginLeft: 0,
          marginRight: 40,
          backgroundColor: 'transparent'
        },
        exporting: {
          url: '/export',
          enabled: !window.isMobile,
          buttons: {
            contextButton: {
              enabled: false
            },
            exportButton: {
              contextButton: {
                enabled: false
              },
              symbol: window.isEN ? 'url(/image/en-save-img-btn.jpg)' : 'url(/image/save-img-btn.png)',
              menuItems: ['downloadPNG', 'downloadJPEG', 'downloadSVG'],
              y: 8,
              x: -10,
              width: 100
            }
          },
          filename: indexCode + '_' + ultls.getTodayValue()
        },
        navigation: {
          buttonOptions: {
            symbolSize: 85,
            symbolY: 54,
            symbolX: 110,
            theme: {
              r: 0,
              states: {
                hover: {
                  fill: '#fff'
                },
                select: {
                  fill: '#fff'
                }
              }
            }
          }
        },
        navigator: {
          enabled: false
        },
        lang: {
          rangeSelectorZoom: ''
        },
        credits: {
          enabled: false
        },
        scrollbar: {
          enabled: false
        },
        rangeSelector: {
          inputEnabled: false,
          enabled: false
        },
        xAxis: {
          alternateGridColor: '#F4F4F4',
          showFirstLabel: true,
          showLastLabel: true,
          labels: {
            formatter: function () {
              const findData = arrChartData.find(item => item.id === this.value);
              return findData ? findData.xLabel : this.value;
            }
          }
        },
        yAxis: [
          {
            left: '40',
            labels: {
              formatter: function () {
                return this.value.toFixed(0);
              }
            }
          }
        ],
        series: [
          {
            type: 'line',
            data: arrChartData,
            name: vm.indexName,
            threshold: null,
            lineWidth: 2,
            color: colors[0],
            yAxis: 0,
            fillColor: {
              linearGradient: {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 1
              },
              stops: [
                [0, Highcharts.getOptions().colors[0]],
                [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
              ]
            }
          }
        ],
        title: {
          text: ''
        },
        plotOptions: {
          series: {
            showInLegend: true,
            gapSize: 10000,
            dataGrouping: {
              enabled: false
            }
          }
        },
        tooltip: {
          crosshairs: [true, true],
          shape: 'square',
          split: true,
          shared: true,
          borderColor: '6495ed',
          useHTML: true,
          formatter: function () {
            const arrCnText = showType === 'realtime' ? ['点位', '涨跌'] : ['开盘', '最高', '最低', '收盘', '涨跌幅'];
            const arrEnText = showType === 'realtime' ? ['Value', 'Change'] : ['Open', 'High', 'Low', 'Close', 'Percent'];
            const arrText = window.isEN ? arrEnText : arrCnText;
            const seriesData = this.points[0].point.series;
            const extraData = this.points[0].point.extra;
            let tooltipText = `<p><b>${seriesData.name}</b></p>`;
            tooltipText += `<p>${extraData.date}</p>`;
            if (showType === 'realtime') {
              tooltipText += `<p>${arrText[0]}：${extraData.now.toFixed(2)}</p>`;
              tooltipText += `<p>${arrText[1]}：${extraData.diffLabel}(${extraData.pctChgLabel})</p>`;
            } else {
              tooltipText += `<p>${arrText[0]}：${extraData.open.toFixed(2)}</p>`;
              tooltipText += `<p>${arrText[1]}：${extraData.high.toFixed(2)}</p>`;
              tooltipText += `<p>${arrText[2]}：${extraData.low.toFixed(2)}</p>`;
              tooltipText += `<p>${arrText[3]}：${extraData.close.toFixed(2)}</p>`;
              tooltipText += `<p>${arrText[4]}：${extraData.pctChgLabel}</p>`;
            }
            return tooltipText;
          }
        }
      };

      if (!window.isMobile) {
        timeChart = Highcharts.stockChart('chart1', option);
        timeChart2 = Highcharts.stockChart('chart2', option);
      } else {
        timeChart = Highcharts.stockChart('mchart', option);
      }
    };

    var timeDateFormate = function (data, gap) {
      if (gap == 1) {
        return data;
      }

      if (data && data.length > 0) {
        var newData = [],
          i = 0;
        for (i; i < data.length; i++) {
          if (i == 0) {
            newData.push(data[i]);
          } else {
            if (i % gap == 0) {
              newData.push(data[i]);
            }
          }
        }
      }

      return newData;
    };
    //创建历史行情图表 type: line or area
    var createHistoryChart = function (historyData, type) {
      if (!$.isEmptyObject(timeChart)) {
        timeChart.destroy();
      }
      //timeChart2.destroy();
      //vm.indicatorsValue = 'MACD';

      var option = {
        chart: {
          zoomType: 'x',
          marginTop: window.isMobile ? 0 : 80,
          marginLeft: 0,
          marginRight: 60,
          backgroundColor: 'transparent',
          resetZoomButton: {
            position: {
              align: 'left',
              x: 10
            }
          }
        },
        noData: {
          style: {
            fontWeight: 'bold',
            fontSize: '15px',
            color: '#303030'
          },
          userHTML: true
        },
        exporting: {
          url: '/export',
          enabled: !window.isMobile,
          buttons: {
            contextButton: {
              enabled: false
            },
            exportButton: {
              symbol: window.isEN ? 'url(/image/en-save-img-btn.jpg)' : 'url(/image/save-img-btn.png)',
              //text: window.isEN ? 'Save': '保存',
              menuItems: ['downloadPNG', 'downloadJPEG', 'downloadSVG'],
              y: 8,
              x: -10,
              width: 100
            }
          },
          filename: indexCode + '_' + ultls.getTodayValue()
        },
        navigation: {
          buttonOptions: {
            symbolSize: 85,
            symbolY: 54,
            symbolX: 110,
            //symbolStrokeWidth: 1,

            theme: {
              // 'stroke-width': 1,
              //  stroke: '#d9d9d9',
              r: 0,
              states: {
                hover: {
                  fill: '#fff'
                },
                select: {
                  fill: '#fff'
                }
              }
            }
          }
        },
        navigator: {
          enabled: false
        },
        lang: {
          rangeSelectorZoom: '',
          noData: window.isEN ? 'no data' : '暂无数据'
        },
        mapNavigation: {
          enabled: true,
          enableButtons: false
        },
        credits: {
          enabled: false
        },
        scrollbar: {
          enabled: false
        },
        rangeSelector: {
          buttonTheme: {
            display: 'none'
          },
          //selected: 4,
          inputEnabled: false,
          enabled: false
        },
        xAxis: {
          alternateGridColor: '#F4F4F4',
          showFirstLabel: true,
          showLastLabel: true,
          labels: {}
        },
        yAxis: [
          {
            left: '0',
            height: '80%',
            labels: {
              align: 'left',
              formatter: function () {
                return this.value;
              }
            }
          },
          {
            top: '80%',
            height: '10%',
            labels: {
              enabled: false
            }
          },
          {
            top: '90%',
            height: '10%',
            labels: {
              enabled: false
            }
          }
        ],
        title: {
          text: ''
        },
        plotOptions: {
          series: {
            showInLegend: true,
            turboThreshold: 100000,
            dataGrouping: {
              enabled: false
            }
          }
        },
        tooltip: {
          crosshairs: [true, true],
          split: true,
          borderColor: 'eee',
          //shared: true,
          xDateFormat: '%Y-%m-%d'
        },
        series: [
          {
            type: type,
            id: 'main',
            data: historyData,
            name: vm.indexName,
            threshold: null,
            color: colors[0],
            lineWidth: 2,
            lastVisiblePrice: {
              enabled: true,
              label: {
                enabled: true
              }
            },
            fillColor: {
              linearGradient: {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 1
              },
              stops: [
                [0, Highcharts.getOptions().colors[0]],
                [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
              ]
            },
            tooltip: {
              crosshairs: [true, true],
              // shape: 'square',
              // split: true,
              // shared: true,
              useHTML: true,
              pointFormatter: function () {
                var s =
                  '<div class="chart-tooltip"><b>' +
                  this.series.name +
                  '</b><br/>' +
                  '<span style="this.series.color">' +
                  Highcharts.dateFormat('%Y-%m-%d', this.x) +
                  '</span><br/>';

                if (vm.realtimemarket == 0) {
                  s += '收盘：' + '<span style="this.series.color">' + this.extra[5].toFixed(2) + '</span><br/>';
                } else {
                  s +=
                    '开盘：' +
                    '<span style="this.series.color">' +
                    this.extra[3].toFixed(2) +
                    '</span><br/>' +
                    '高：' +
                    '<span style="this.series.color">' +
                    this.extra[2].toFixed(2) +
                    '</span><br/>' +
                    '低：' +
                    '<span style="this.series.color">' +
                    this.extra[4].toFixed(2) +
                    '</span><br/>' +
                    '收盘：' +
                    '<span style="this.series.color">' +
                    this.extra[5].toFixed(2) +
                    '</span><br/>';
                }

                //涨跌趋势渲染
                if (this.extra[6] > 0) {
                  s += '涨跌：' + '<span>+' + this.extra[6].toFixed(2) + '(+' + (this.extra[7] * 100).toFixed(2) + '%)' + '</span><br/>';
                } else if (this.extra[6] == null) {
                  s += '涨跌：' + '<span>' + '--' + '(' + '--' + '%)' + '</span><br/>';
                } else {
                  s += '涨跌：' + '<span>' + this.extra[6].toFixed(2) + '(' + (this.extra[7] * 100).toFixed(2) + '%)' + '</span><br/>';
                }
                s +=
                  '成交额：' + '<span>' + (!vm.arrEmptyData.includes(this.extra[8]) ? ultls.moneyUnits(this.extra[8]) + '元' : '--') + '</span><br/>';
                // 债券指数
                if (vm.newIndexType) {
                  s +=
                    '成交量：' + '<span>' + (!vm.arrEmptyData.includes(this.extra[9]) ? ultls.volumeBondUnits(this.extra[9]) : '--') + '</span><br/>';
                } else {
                  // 非债券指数
                  s += '成交量：' + '<span>' + (!vm.arrEmptyData.includes(this.extra[9]) ? ultls.volumeUnits(this.extra[9]) : '--') + '</span><br/>';
                }
                var sEN =
                  '<b>' +
                  this.series.name +
                  '</b><br/>' +
                  '<span style="this.series.color">' +
                  Highcharts.dateFormat('%Y-%m-%d', this.x) +
                  '</span><br/>';
                if (vm.realtimemarket == 0) {
                  sEN += 'Close：' + '<span style="this.series.color">' + this.extra[5].toFixed(2) + '</span><br/>';
                } else {
                  sEN +=
                    'Open：' +
                    '<span style="this.series.color">' +
                    this.extra[3].toFixed(2) +
                    '</span><br/>' +
                    'High：' +
                    '<span style="this.series.color">' +
                    this.extra[2].toFixed(2) +
                    '</span><br/>' +
                    'Low：' +
                    '<span style="this.series.color">' +
                    this.extra[4].toFixed(2) +
                    '</span><br/>' +
                    'Close：' +
                    '<span style="this.series.color">' +
                    this.extra[5].toFixed(2) +
                    '</span><br/>';
                }
                //涨跌趋势渲染
                if (this.extra[6] > 0) {
                  sEN += 'Change：' + '<span>+' + this.extra[6].toFixed(2) + '(+' + (this.extra[7] * 100).toFixed(2) + '%)' + '</span><br/>';
                } else if (this.extra[6] == null) {
                  sEN += 'Change：' + '<span>' + '--' + '(' + '--' + '%)' + '</span><br/>';
                } else {
                  sEN += 'Change：' + '<span>' + this.extra[6].toFixed(2) + '(' + (this.extra[7] * 100).toFixed(2) + '%)' + '</span><br/>';
                }
                sEN += 'Turnover：' + '<span>' + (!vm.arrEmptyData.includes(this.extra[8]) ? ultls.moneyUnits(this.extra[8]) : '--') + '</span><br/>';
                //债券指数
                if (vm.newIndexType) {
                  sEN +=
                    'Volume：' + '<span>' + (!vm.arrEmptyData.includes(this.extra[9]) ? ultls.volumeBondUnits(this.extra[9]) : '--') + '</span><br/>';
                } else {
                  //非债券指数
                  sEN +=
                    'Volume：' + '<span>' + (!vm.arrEmptyData.includes(this.extra[9]) ? ultls.volumeUnits(this.extra[9]) : '--') + '</span><br/>';
                }
                return window.isEN ? sEN : s;
              }
            }
          },
          {
            type: 'column',
            id: 'column',
            data: volumeData,
            name: vm.indexName,
            yAxis: 1,
            tooltip: {
              pointFormatter: function () {
                return '';
              }
            }
          },
          macdOption
        ]
      };
      if (!window.isMobile) {
        timeChart = Highcharts.stockChart('chart1', option);
        timeChart2 = Highcharts.stockChart('chart2', option);
      } else {
        timeChart = Highcharts.stockChart('mchart', option);
      }
    };
    //创建对比指数行情图表 type: line
    var createContrastChart = function (contrastData, type = 'line') {
      var seriesList = [],
        dataList = [];
      if (contrastData.length > 0) {
        var i = 0;
        for (i; i < contrastData.length; i++) {
          var currentData = contrastData[i].dataList,
            returnList = [];
          currentData.forEach(function (item) {
            var dataItem = {
              x: item[0], // 横坐标用时间戳
              y: item[11] * 100, // 竖坐标用数据中的第11个值
              extra: item
            };
            returnList.push(dataItem);
          });
          var obj = {
            type: type,
            id: contrastData[i].indexCode,
            data: returnList,
            color: colors[i],
            name: window.isEN ? contrastData[i].indexEName : contrastData[i].indexName,
            threshold: null,
            lineWidth: 2,
            fillColor: {
              linearGradient: {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 1
              },
              stops: [
                [0, Highcharts.getOptions().colors[0]],
                [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
              ]
            },
            tooltip: {
              crosshairs: [true, true],
              //split: true,
              //shared: true,
              xDateFormat: '%Y-%m-%d',
              pointFormatter: function () {
                var s =
                  '<b>' +
                  this.series.name +
                  '</b><br/>' +
                  '<span style="this.series.color">' +
                  Highcharts.dateFormat('%Y-%m-%d', this.x) +
                  '</span><br/>' +
                  '点位：' +
                  '<span style="this.point.series.color">' +
                  this.extra[5].toFixed(2) +
                  '</span><br/>';
                // 涨跌
                if (this.extra[7] > 0) {
                  s += '涨跌：' + '<span>+' + (this.extra[7] * 100).toFixed(2) + '%' + '</span><br/>';
                } else {
                  s += '涨跌：' + '<span>' + (this.extra[7] * 100).toFixed(2) + '%' + '</span><br/>';
                }
                if (vm.arrEmptyData.includes(this.extra[8])) {
                  s += '成交额：--<br/>';
                } else {
                  s += '成交额：' + '<span>' + ultls.moneyUnits(this.extra[8]) + '元</span><br/>';
                }
                if (vm.arrEmptyData.includes(this.extra[9])) {
                  s += '成交量：--<br/>';
                } else {
                  s += '成交量：' + '<span>' + ultls.volumeUnits(this.extra[9]) + '</span><br/>';
                }
                var sEN =
                  '<b>' +
                  this.series.name +
                  '</b><br/>' +
                  '<span style="this.series.color">' +
                  Highcharts.dateFormat('%Y-%m-%d', this.x) +
                  '</span><br/>' +
                  'Value：' +
                  '<span style="this.point.series.color">' +
                  this.extra[5].toFixed(2) +
                  '</span><br/>';
                // 英文涨跌
                if (this.extra[7] > 0) {
                  sEN += 'Change：' + '<span>+' + (this.extra[7] * 100).toFixed(2) + '%' + '</span><br/>';
                } else {
                  sEN += 'Change：' + '<span>' + (this.extra[7] * 100).toFixed(2) + '%' + '</span><br/>';
                }
                if (vm.arrEmptyData.includes(this.extra[8])) {
                  sEN += 'Volume：--<br/>';
                } else {
                  sEN += 'Volume：' + '<span>' + ultls.moneyUnits(this.extra[8]) + '</span><br/>';
                }
                if (vm.arrEmptyData.includes(this.extra[9])) {
                  sEN += 'Turnover：--<br/>';
                } else {
                  sEN += 'Turnover：' + '<span>' + ultls.volumeUnits(this.extra[9]) + '</span><br/>';
                }
                return window.isEN ? sEN : s;
              }
            }
          };
          seriesList.push(obj);
        }
      }
      var option = {
        chart: {
          zoomType: 'x',
          marginTop: window.isMobile ? 0 : 80,
          marginLeft: 0,
          marginRight: 80,
          backgroundColor: 'transparent'
        },
        noData: {
          style: {
            fontWeight: 'bold',
            fontSize: '15px',
            color: '#303030'
          }
        },
        exporting: {
          url: '/export',
          enabled: !window.isMobile,
          buttons: {
            contextButton: {
              enabled: false
            },
            exportButton: {
              symbol: window.isEN ? 'url(/image/en-save-img-btn.jpg)' : 'url(/image/save-img-btn.png)',
              //text: window.isEN ? 'Save': '保存',
              menuItems: ['downloadPNG', 'downloadJPEG', 'downloadSVG'],
              y: 8,
              x: -10,
              width: 100
            }
          },
          filename: indexCode + '_' + ultls.getTodayValue()
        },
        navigation: {
          buttonOptions: {
            symbolSize: 85,
            symbolY: 54,
            symbolX: 110,
            //symbolStrokeWidth: 1,

            theme: {
              // 'stroke-width': 1,
              //  stroke: '#d9d9d9',
              r: 0,
              states: {
                hover: {
                  fill: '#fff'
                },
                select: {
                  fill: '#fff'
                }
              }
            }
          }
        },
        navigator: {
          enabled: false
        },
        lang: {
          rangeSelectorZoom: ''
        },
        mapNavigation: {
          enabled: true,
          enableButtons: false
        },
        credits: {
          enabled: false
        },
        scrollbar: {
          enabled: false
        },
        rangeSelector: {
          buttonTheme: {
            display: 'none'
          },
          inputEnabled: false,
          enabled: false
        },
        xAxis: {
          alternateGridColor: '#F4F4F4',
          showFirstLabel: true,
          showLastLabel: true,
          labels: {}
        },
        yAxis: [
          {
            left: '10',
            height: '100%',
            labels: {
              align: 'left',
              formatter: function () {
                return this.value > 0 ? '+' + this.value + '%' : this.value + '%';
              }
            }
          }
        ],
        title: {
          text: ''
        },
        plotOptions: {
          series: {
            //compare: 'percent'
            // showInLegend: true,
            turboThreshold: 0,
            dataGrouping: {
              enabled: true
            }
          }
        },
        series: seriesList
      };

      if (!window.isMobile) {
        timeChart = Highcharts.stockChart('chart1', option);
        timeChar2 = Highcharts.stockChart('chart2', option);
      } else {
        timeChart = Highcharts.stockChart('mchart', option);
      }
    };
    // 创建K线图
    var createKlineChart = function (klineData) {
      timeChart.destroy();
      var option = {
        chart: {
          zoomType: 'x',
          marginTop: window.isMobile ? 0 : 80,
          marginLeft: 0,
          marginRight: 60,
          backgroundColor: 'transparent'
        },
        exporting: {
          url: '/export',
          enabled: !window.isMobile,
          buttons: {
            contextButton: {
              enabled: false
            },
            exportButton: {
              symbol: window.isEN ? 'url(/image/en-save-img-btn.jpg)' : 'url(/image/save-img-btn.png)',
              // text: window.isEN ? 'Save': '保存',
              menuItems: ['downloadPNG', 'downloadJPEG', 'downloadSVG'],
              y: 8,
              x: -10,
              width: 100
            }
          },
          filename: indexCode + '_' + ultls.getTodayValue()
        },
        navigation: {
          buttonOptions: {
            symbolSize: 85,
            symbolY: 54,
            symbolX: 110,
            //symbolStrokeWidth: 1,
            theme: {
              // 'stroke-width': 1,
              //  stroke: '#d9d9d9',
              r: 0,
              states: {
                hover: {
                  fill: '#fff'
                },
                select: {
                  fill: '#fff'
                }
              }
            }
          }
        },
        navigator: {
          enabled: false
        },
        mapNavigation: {
          enabled: true,
          enableButtons: false
        },
        lang: {
          rangeSelectorZoom: ''
        },
        credits: {
          enabled: false
        },
        rangeSelector: {
          buttonTheme: {
            display: 'none'
          },
          inputEnabled: false,
          enabled: false
        },
        scrollbar: {
          enabled: false
        },
        xAxis: {
          alternateGridColor: '#F4F4F4',
          showFirstLabel: true,
          showLastLabel: true,
          labels: {}
        },
        yAxis: [
          {
            left: '40',
            height: '80%'
            // labels: {
            //     formatter: function () {
            //         return this.value.toFixed(2)
            //     }
            // }
          },
          {
            left: '40',
            top: '80%',
            height: '10%',
            labels: {
              enabled: false
            }
          },
          {
            left: '40',
            top: '90%',
            height: '10%',
            labels: {
              enabled: false
            }
          }
        ],
        title: {
          text: ''
        },
        plotOptions: {
          series: {
            showInLegend: true,
            turboThreshold: 100000,
            dataGrouping: {
              enabled: false
            }
          }
        },
        tooltip: {
          shape: 'square',
          split: true,
          shared: true,
          borderColor: '#fff'
        },
        series: [
          {
            id: 'main',
            type: 'candlestick',
            data: klineData,
            name: vm.indexName,
            threshold: null,
            color: '#059825',
            colorByPoint: true,
            lineColor: '#059825',
            upColor: '#ff2e28',
            upLineColor: '#ff2e28',
            pointPadding: 0.2,
            tooltip: {
              pointFormatter: function () {
                var s =
                  '<div class="chart-tooltip"><b>' +
                  this.series.name +
                  '</b><br/>' +
                  '<span style="this.series.color">' +
                  Highcharts.dateFormat('%Y-%m-%d', this.x) +
                  '</span><br/>' +
                  '开盘：' +
                  '<span style="this.series.color">' +
                  this.extra[3].toFixed(2) +
                  '</span><br/>' +
                  '高：' +
                  '<span style="this.series.color">' +
                  this.extra[2].toFixed(2) +
                  '</span><br/>' +
                  '低：' +
                  '<span style="this.series.color">' +
                  this.extra[4].toFixed(2) +
                  '</span><br/>' +
                  '收盘：' +
                  '<span style="this.series.color">' +
                  this.extra[5].toFixed(2) +
                  '</span><br/>';
                //涨跌趋势渲染
                if (this.extra[6] > 0) {
                  s += '涨跌：' + '<span>+' + this.extra[6].toFixed(2) + '(+' + (this.extra[7] * 100).toFixed(2) + '%)' + '</span><br/>';
                } else {
                  s += '涨跌：' + '<span>' + this.extra[6].toFixed(2) + '(' + (this.extra[7] * 100).toFixed(2) + '%)' + '</span><br/>';
                }
                s +=
                  '成交额：' + '<span>' + (!vm.arrEmptyData.includes(this.extra[8]) ? ultls.moneyUnits(this.extra[8]) + '元' : '--') + '</span><br/>';
                s +=
                  '成交量：' + '<span>' + (!vm.arrEmptyData.includes(this.extra[9]) ? ultls.volumeUnits(this.extra[9]) : '--') + '</span><br/></div>';

                var sEN =
                  '<div class="chart-tooltip"><b>' +
                  this.series.name +
                  '</b><br/>' +
                  '<span style="this.series.color">' +
                  Highcharts.dateFormat('%Y-%m-%d', this.x) +
                  '</span><br/>' +
                  'Open：' +
                  '<span style="this.series.color">' +
                  this.extra[3].toFixed(2) +
                  '</span><br/>' +
                  'High：' +
                  '<span style="this.series.color">' +
                  this.extra[2].toFixed(2) +
                  '</span><br/>' +
                  'Low：' +
                  '<span style="this.series.color">' +
                  this.extra[4].toFixed(2) +
                  '</span><br/>' +
                  'Close：' +
                  '<span style="this.series.color">' +
                  this.extra[5].toFixed(2) +
                  '</span><br/>';
                //涨跌趋势渲染
                if (this.extra[6] > 0) {
                  sEN += 'Change：' + '<span>+' + this.extra[6].toFixed(2) + '(+' + (this.extra[7] * 100).toFixed(2) + '%)' + '</span><br/>';
                } else {
                  sEN += 'Change：' + '<span>' + this.extra[6].toFixed(2) + '(' + (this.extra[7] * 100).toFixed(2) + '%)' + '</span><br/>';
                }
                sEN += 'Volume：' + '<span>' + (!vm.arrEmptyData.includes(this.extra[8]) ? ultls.moneyUnits(this.extra[8]) : '--') + '</span><br/>';
                sEN +=
                  'Turnover：' +
                  '<span>' +
                  (!vm.arrEmptyData.includes(this.extra[9]) ? ultls.volumeUnits(this.extra[9]) : '--') +
                  '</span><br/></div>';

                return window.isEN ? sEN : s;
              }
            }
          },
          {
            type: 'column',
            id: 'column',
            data: volumeData,
            name: vm.indexName,
            yAxis: 1,
            tooltip: {
              pointFormatter: function () {
                return '';
              }
            }
          },
          macdOption
        ]
      };
      if (!window.isMobile) {
        timeChart = Highcharts.stockChart('chart1', option);
        timeChar2 = Highcharts.stockChart('chart2', option);
      } else {
        timeChart = Highcharts.stockChart('mchart', option);
      }
    };

    // 线下指数查询历史行情数据
    var queryIndexHistoryData = function (code) {
      var historyData = [];
      var requestUrlList = {
        day: '/market/market/getIndexDailyData',
        week: '/market/market/getIndexWeekData',
        month: '/market/market/getIndexMonthData',
        year: '/market/market/getIndexYearData'
      };
      if (vm) {
        var requestUrl = requestUrlList[vm.chartFrequencyType];
      } else {
        var requestUrl = requestUrlList['month'];
      }
      // 如果时间范围为空，则限制在基日开始日期和基日结束日期之间
      const datePeriod = vm.startAndEndTime && vm.startAndEndTime.length > 0 ? vm.startAndEndTime : [vm.jrStartDate, vm.jrEndDate];
      $.ajax({
        url: window.hqUrl + requestUrl,
        type: 'get',
        data: {
          indexCode: code,
          startDate: datePeriod[0],
          endDate: datePeriod[1]
        },
        dataType: 'json',
        success: function (res) {
          if (res.code == 200) {
            var i = 0;
            historyData = res.data.data;
            ohclData = [];
            newHistoryData = [];
            volumeData = [];
            if (historyData && historyData.length > 0) {
              for (i; i < historyData.length; i++) {
                var obj = {
                  x: historyData[i][0],
                  y: historyData[i][5],
                  extra: historyData[i]
                };
                newHistoryData.push(obj);
                if (historyData[i][3] == historyData[i][5]) {
                  ohclData.push({
                    x: historyData[i][0], //the date
                    open: historyData[i][3], //open
                    high: historyData[i][2], //high
                    low: historyData[i][4], //low
                    close: historyData[i][5] + (historyData[i][6] > 0 ? 1 : -1), //close
                    extra: historyData[i]
                  });
                } else {
                  ohclData.push({
                    x: historyData[i][0], //the date
                    open: historyData[i][3], //open
                    high: historyData[i][2], //high
                    low: historyData[i][4], //low
                    close: historyData[i][5], //close
                    extra: historyData[i]
                  });
                }
                volumeData.push({
                  x: historyData[i][0], //the date
                  y: historyData[i][9], //volume
                  color: historyData[i][6] < 0 ? 'green' : 'red'
                });
              }
              // 绘制线图
              vm.showStockChart = true;
              if (vm.currentChartType == 'line' || vm.currentChartType == 'area') {
                // 绘制折线、面积图
                createHistoryChart(newHistoryData, vm.currentChartType);
              } else {
                createKlineChart(ohclData);
              }
            } else {
              vm.showStockChart = false;
            }
          } else {
            vm.showStockChart = false;
          }
        },
        error: function (e) {
          vm.showStockChart = false;
        }
      });
    };

    // 线下指数查询实时行情数据
    var queryOfflineRealtimeData = function (code) {
      $.ajax({
        url: window.hqUrl + '/market/market/getIndexLatestRealTimeData',
        type: 'get',
        async: true,
        data: { indexCode: code, t: new Date().getTime() },
        dataType: 'json'
      }).done(function (res) {
        // 赋值实时行情数据
        if (res.code === 200 && res.data && Array.isArray(res.data.data) && res.data.data.length > 0) {
          const arrData = res.data.data || [];
          const firstData = arrData[0] || {};
          vm.indexCurrent = firstData[1] ? firstData[1].toFixed(2) : '--';
          vm.indexChg = firstData[6] ? firstData[6].toFixed(2) : '--';
          vm.indexPercent = firstData[7] ? (firstData[7] * 100).toFixed(2) : '';
          vm.indexTime = ultls.formateDateTime(firstData[0]);
          vm.indexMoney = firstData[8] ? ultls.moneyUnits(firstData[8]) : '--';
        } else {
          vm.indexCurrent = '--';
          vm.indexChg = '--';
          vm.indexPercent = '';
          vm.indexTime = '--';
          vm.indexMoney = '--';
        }
        // 更新移动端顶部交易时间
        vm.setDomText({ domId: 'mobIndexTime', text: vm.indexTime });
      });
    };
    // 线下指数查询分时图数据
    var queryOfflineChartData = function (code) {
      $.ajax({
        url: window.hqUrl + '/market/market/getIndexRealTimeData',
        type: 'get',
        data: { indexCode: code },
        dataType: 'json'
      }).done(function (res) {
        vm.showStockChart = true;
        timeData = [];
        if (res.code == 200) {
          var newData = res.data.data;
          for (var i = 0; i < newData.length; i++) {
            if (i == 0) {
              preClosePrice = newData[0][11];
            }
            var obj = {
              x: newData[i][0],
              y: newData[i][1],
              extra: newData[i]
            };
            timeData.push(obj);
          }
          createTimeChart(timeData, vm.currentChartType, vm.chartFrequencyType, preClosePrice);
        } else {
          createTimeChart(timeData, vm.currentChartType, vm.chartFrequencyType, preClosePrice);
        }
      });
    };
    var queryIndexConfigInfo = function (code) {
      $.ajax({
        url: '/index/selectIndexByCode',
        type: 'get',
        async: true,
        data: { codeValue: code, time: new Date().getTime() },
        dataType: 'json'
      }).done(function (res) {
        if (res.code !== 200 || !res.data) {
          return;
        }
        const resData = res.data;
        vm.indexType = resData.indextype;
        vm.showDetail = resData.showdetail;
        vm.indexName = window.isEN ? resData.indexename : resData.indexname;
        vm.indexCodeName = resData.indexcode;
        vm.realtimemarket = resData.realtimemarket;
        vm.isRealtimeMarket = Number(resData.realtimemarket) === 1; // 是否有实时行情
        vm.indexSource = resData.indexsource;
        vm.isOnlineIndex = Number(resData.indexsource) === 1; // 是否线上指数
        vm.dataSource = resData.dataSource || 0; // 指数数据来源
        // 更新移动端顶部的指数名称
        vm.setDomText({ domId: 'mobIndexName', text: vm.indexName });
        // 更新基日开始时间
        if (resData.jr) {
          vm.jrStartDate = resData.jr;
        }

        // 全球指数不展示相关产品
        if (vm.indexType == 700) {
          vm.tabsList = window.isEN
            ? [
                { name: 'Index Performance', id: '0' },
                { name: 'Historical Data', id: '1' },
                { name: 'Constituents', id: '2' }
              ]
            : [
                { name: '指数表现', id: '0' },
                { name: '历史行情', id: '1' },
                { name: '样本详情', id: '2' }
              ];
        } else {
          vm.tabsList = window.isEN
            ? [
                { name: 'Index Performance', id: '0' },
                { name: 'Historical Data', id: '1' },
                { name: 'Constituents', id: '2' },
                { name: 'Related Products', id: '3' }
              ]
            : [
                { name: '指数表现', id: '0' },
                { name: '历史行情', id: '1' },
                { name: '样本详情', id: '2' },
                { name: '相关产品', id: '3' }
              ];
        }
        if (vm.showDetail && vm.showDetail == '0') {
          vm.tabsList.forEach(function (item, index) {
            if (item.name == 'Constituents' || item.name == '样本详情') {
              vm.tabsList.splice(index, 1);
            }
          });
          vm.mobileTabsList.forEach(function (item, index) {
            if (item.name == 'Sample' || item.name == '样本') {
              vm.mobileTabsList.splice(index, 1);
            }
          });
        }
        // 请求相关产品
        if (vm.indexType != 700) {
          vm.queryFundList(1, 20);
        }

        if (window.isEN) {
          vm.docDownload =
            '/en/module/pdf-detail.html?pdf=' + '/docs/gz_' + indexCode + '_e.pdf&name=' + vm.indexName + '&indexCode=' + indexCode + '&type=1';
          vm.colorPageDownload =
            '/en/module/pdf-detail-pro.html?pdf=' + '/docs/jj_' + indexCode + '_e.pdf&name=' + vm.indexName + '&indexCode=' + indexCode + '&type=2';
        } else {
          vm.docDownload =
            '/module/pdf-detail.html?pdf=' + '/docs/gz_' + indexCode + '.pdf&name=' + vm.indexName + '&indexCode=' + indexCode + '&type=1';
          vm.colorPageDownload =
            '/module/pdf-detail-pro.html?pdf=' + '/docs/jj_' + indexCode + '.pdf&name=' + vm.indexName + '&indexCode=' + indexCode + '&type=2';
        }
        // 如果是线上指数，一定会有实时行情，请求NSDK获取实时行情数据
        if (vm.isOnlineIndex) {
          vm.queryNsdkRealTimeData();
          vm.setUpdateEveryFiveMinute();
        } else if (vm.isRealtimeMarket) {
          // 线下指数、有实时行情，查实时行情、查分时图
          queryOfflineRealtimeData(indexCode);
          queryOfflineChartData(indexCode);
          vm.setUpdateEveryFiveMinute();
        } else {
          // 线下指数、没有实时行情，切到1M，查询近一个月数据
          vm.dateOption = vm.dateOptionPro;
          vm.dateOptionValue = '1M';
          vm.startAndEndTime = [ultls.getPreMonthDay(), ultls.getTodayValue()];
          vm.chartFrequencyType = 'day';
          queryIndexHistoryData(indexCode);
        }
      });
    };
    // 判断当前指数是不是新版债券指数
    var queryJudgeBondIndexType = function (code) {
      $.ajax({
        url: '/bondIndexSample/judgeBondIndexType',
        type: 'get',
        async: true,
        data: { indexcode: code, t: new Date().getTime() },
        dataType: 'json'
      }).done(function (res) {
        if (res.code == 200) {
          vm.newIndexType = res.data;
        }
      });
    };
    //实例化 vm
    var vm = new Vue({
      el: '#index',
      data: function () {
        return {
          showHide: true,
          allOnlineIndexList: [], // 所有线上指数集合
          isOnlineIndex: false,
          jrStartDate: '1991-04-19', // 查询的基础开始日期
          jrEndDate: '2099-01-01', // 查询的基础结束日期
          indexSource: '', //指数来源（1、线上 0、线下）
          // 实时行情延迟到15:30的几个指数
          moreTimeIndexList: [
            { indexCode: '399301', indexName: '深信用债' },
            { indexCode: '399302', indexName: '深公司债' },
            { indexCode: '399298', indexName: '深信中高' },
            { indexCode: '399299', indexName: '深信中低' },
            { indexCode: '399290', indexName: '深转交债' },
            { indexCode: '399481', indexName: '企债指数' }
          ],
          indexTradeInfo: {
            isDone: false,
            isTradeDay: true,
            lastTradeDay: ''
          },
          isFix: false,
          isEN: window.isEN,
          colors: colors,
          isContrast: false, //是否为对比指数
          isLoading: true,
          activeName: '0',
          dataValue: '',
          hideSampleList: [
            '399312',
            '399361',
            '399365',
            '399366',
            '399412',
            '399423',
            '399432',
            '399436',
            '399437',
            '399438',
            '399439',
            '980015',
            '980032',
            '980033',
            'CN2312',
            'CN2361',
            'CN2365',
            'CN2366',
            'CN2412',
            'CN2423',
            'CN2432',
            'CN2436',
            'CN2437',
            'CN2438',
            'CN2439',
            '480015',
            '480032',
            '480033',
            'CN5074',
            'CN6074',
            'CN5079',
            'CN6079'
          ],
          tableData: [],
          startAndEndTime: [],
          startAndEndTimeHistory: [],
          datePickValue: '',
          //标签栏
          tabsList: [],
          //移动端标签页
          mobileTabsList: window.isEN
            ? [
                { name: 'Survey', id: '0' },
                { name: 'Performance', id: '1' },
                { name: 'Market', id: '2' },
                { name: 'Sample', id: '3' },
                { name: 'Product', id: '4' }
              ]
            : [
                { name: '概况', id: '0' },
                { name: '表现', id: '1' },
                { name: '行情', id: '2' },
                { name: '样本', id: '3' },
                { name: '产品', id: '4' }
              ],
          //是否为移动端
          isMobile: window.isMobile,
          //初始化图表类型
          currentChartType: 'line',
          //默认图表类型
          chartTypeList: window.isEN
            ? {
                line: 'line',
                candlestick: 'candle...',
                area: 'area'
              }
            : {
                line: '折线图',
                candlestick: 'K线图',
                area: '面积图'
              },
          //图表频次
          chartFrequencyType: 'm1',
          chartFrequencyList: window.isEN
            ? {
                m1: '1min',
                m15: '15min',
                m30: '30min',
                m60: '60min',
                day: 'day',
                week: 'week',
                month: 'month',
                year: 'year'
              }
            : {
                m1: '1min',
                m15: '15min',
                m30: '30min',
                m60: '60min',
                day: '日',
                week: '周',
                month: '月',
                year: '年'
              },
          //默认日期区间（有实时行情）
          dateOptionValue: '1D',
          dateOption: ['1D', '1M', '6M', '1Y', 'ALL'],
          //默认日期区间(无实时行情）
          dateOptionPro: ['1M', '6M', '1Y', 'ALL'],
          //历史行情 日期
          dateOptionHistory: '1M',
          //历史行情 默认频次
          historyFrequencyType: 'day',
          //历史行情 全量数据
          historyLoading: false,
          historyPage: 1,
          historyAllData: [],
          historyShowData: [],
          historyAllDataTotal: 0,
          //指数详情数据
          indexName: '', //指数名
          indexCodeName: indexCode, //指数代码
          indexCurrent: '', //当前价格
          indexChg: '', //涨跌额
          indexPercent: '', //涨跌幅
          indexTime: '', //交易时间
          indexMoney: '', //交易额
          //相关基金产品
          fundLoading: false,
          fundList: [],
          fundUpdateTime: '',
          fundPage: 1,
          fundTotal: 0,
          //样本详情
          sampleDate: sampleShow ? ultls.getPrePreMonth() : ultls.getPreMonth(),
          currDate: ultls.getCurrMonth(),
          samplePage: 1,
          sampleTotal: 0,
          sampleLoading: false,
          sampleList: [],
          sampleShow: sampleShow,
          indexType: '', // 指数类型，106、206为债券指数
          arrEmptyData: ['', null, undefined, NaN],
          //指数简介、指数编制
          indexInfo: {},
          //饼图更新时间
          marketDate: '',
          tradeDate: '',
          //阶段性收益数据列表
          incomeDate: '',
          incomeList: [],
          // 调整分析指标
          indexAnalysis: [],
          // 是否显示调整分析指标
          indexAnalysisFlag: false,
          //指数对比
          //键盘精灵输入框
          searchVal: '',
          //对比函数列表
          contrastIndexList: [],
          queryContrastList: [],
          //全屏状态
          fullScreenDialog: false,
          //相关新闻列表
          newsList: [],
          docDownload: '',
          docDownloadPDF: '',
          colorPageDownload: '',
          colorPageDownloadPDF: '',
          indicatorsValue: 'MACD',
          //判断是否有实时行情(0 为无实时行情)
          realtimemarket: 1,
          //对比指数
          showHotIndexBox: false,
          // 指数数据来源：0=系统指数，1=外部合作指数
          dataSource: 0,
          //热门指数列表
          hotIndexList: hotIndexList,
          //调样周期
          typlList: window.isEN
            ? {
                1: 'Annually',
                2: 'Semi-annually',
                4: 'Quarterly',
                12: 'Monthly',
                24: 'Semi-monthly',
                0: 'Irregularly'
              }
            : {
                1: '每年一次',
                2: '每半年一次',
                4: '每季度一次',
                12: '每月一次',
                24: '每半个月一次',
                0: '不定期'
              },
          //Echart 图表是否展示
          showEchart: true,
          //highstock 图表是否展示
          showStockChart: true,
          //时间选择组件
          pickOptions: {
            disabledDate: function (time) {
              return time.getTime() > Date.now() - 8.64e6;
            }
          },
          //移动端所需参数
          mobileActiveName: '0',
          mobileFullScreenDialog: false,
          //新版债券指数
          newIndexType: '',
          showDetail: '', //样本详情是否展示 "1"：显示  "0"：不显示
          // 两个特殊恒生指数的相应数据
          hsiIndexData: {
            // 通用数据
            common: {
              realtimeUrl: '/market/HSI/getRealMarket', // 实时行情接口
              historyUrl: '/market/HSI/getHistory', // 历史行情列表接口
              historyDownUrl: '/market/HSI/downloadHistoryExcel', // 历史行情下载接口
              sampleUrl: '/market/HSI/getSamples', // 样本详情接口
              sampleDownUrl: '/market/HSI/downloadSamplesExcel' // 样本详情下载接口
            },
            // 国证恒生大湾区数字经济指数
            SZHKDE: {
              // 指数简介和指数编制数据
              cnIndexInfo: {
                jsjj: '国证恒生大湾区数字经济指数反映与数字经济业务相关的大湾区深港上市公司的表现。',
                xyfw: '深港通标的范围内的粤港澳大湾区上市公司',
                xygz: '公司市值排名最高的 50 间公司会被选为成份股公司',
                jsfs: '流通市值',
                qzsx: 0.1,
                typl: 4,
                jd: '3000',
                jr: '2018-12-31',
                fbrq: '2025-5-19'
              },
              enIndexInfo: {
                jsjj: 'Hang Seng CNI GBA Digital Economy Index aims to gauge the performance of Greater Bay Area companies that are involved in digital economy business and listed in Shenzhen or Hong Kong.',
                xyfw: 'A-shares listed on Shenzhen Stock Exchange and securities listed on HKEX that are eligible for Northbound or Southbound trading under the Stock Connect Scheme',
                xygz: 'Top 50 companies by Company MV Rank are selected',
                jsfs: 'Freefloat-adjusted market capitalisation',
                qzsx: 0.1,
                typl: 4,
                jd: '3000',
                jr: '2018-12-31',
                fbrq: '2025-5-19'
              },
              // 中文资料下载地址
              cnMaterialUrl: {
                scheme: '../../files/gzhszs/SZHKDE_bzfa_cn.pdf',
                colorPage: '../../files/gzhszs/SZHKDE_cy_cn.pdf'
              },
              // 英文资料下载地址
              enMaterialUrl: {
                scheme: '../../files/gzhszs/SZHKDE_bzfa_en.pdf',
                colorPage: '../../files/gzhszs/SZHKDE_cy_en.pdf'
              }
            },
            // 国证恒生大湾区消费指数
            SZHKCON: {
              // 指数简介和指数编制数据
              cnIndexInfo: {
                jsjj: '国证恒生大湾区消费指数反映提供与日常消费相关的消费品制造及服务的大湾区深港上市公司的表现。',
                xyfw: '深港通标的范围内的粤港澳大湾区上市公司',
                xygz: '公司市值排名最高的 50 间公司会被选为成份股公司',
                jsfs: '流通市值',
                qzsx: 0.1,
                typl: 4,
                jd: '3000',
                jr: '2018-12-31',
                fbrq: '2025-5-19'
              },
              enIndexInfo: {
                jsjj: 'Hang Seng CNI GBA Consumption Index aims to gauge the performance of Greater Bay Area companies that provide goods and services relating to daily consumption and listed in Shenzhen or Hong Kong.',
                xyfw: 'A-shares listed on Shenzhen Stock Exchange and securities listed on HKEX that are eligible for Northbound or Southbound trading under the Stock Connect Scheme',
                xygz: 'Top 50 companies by Company MV Rank are selected',
                jsfs: 'Freefloat-adjusted market capitalisation',
                qzsx: 0.1,
                typl: 4,
                jd: '3000',
                jr: '2018-12-31',
                fbrq: '2025-5-19'
              },
              // 中文资料下载地址
              cnMaterialUrl: {
                scheme: '../../files/gzhszs/SZHKCON_bzfa_cn.pdf',
                colorPage: '../../files/gzhszs/SZHKCON_cy_cn.pdf'
              },
              // 英文资料下载地址
              enMaterialUrl: {
                scheme: '../../files/gzhszs/SZHKCON_bzfa_en.pdf',
                colorPage: '../../files/gzhszs/SZHKCON_cy_en.pdf'
              }
            }
          },
          932604: {
            // 中文资料下载地址
            cnMaterialUrl: {
              colorPage: '../../files/zzzs/932604_cy_cn.pdf'
            },
            // 英文资料下载地址
            enMaterialUrl: {
              colorPage: '../../files/zzzs/932604_cy_en.pdf'
            }
          }
        };
      },
      watch: {
        // 监听指数对比的数组
        contrastIndexList: function (value) {
          // 如果有数据，证明打开了指数对比
          if (this.contrastIndexList.length > 0) {
            var i = 0,
              str = indexCode + ';';
            for (i; i < this.contrastIndexList.length; i++) {
              str += this.contrastIndexList[i].indexCode + ';';
            }
            if (!this.isContrast) {
              this.startAndEndTime = [ultls.getPreMonthDay(), ultls.getTodayValue()];
              this.dateOptionValue = '1M';
            }
            this.chartFrequencyType = 'day';
            this.currentChartType = 'line';
            queryIndexContrastData(str);
            this.isContrast = true;
          } else {
            // 没数据，则是关闭了指数对比
            this.isContrast = false;
            // 如果是线上指数 或者 有实时行情的线下指数，日期区间恢复到1D、时间范围清空、频次切到m1
            if (this.isOnlineIndex || this.isRealtimeMarket) {
              this.dateOptionValue = '1D';
              this.startAndEndTime = [];
              this.chartFrequencyType = 'm1';
              this.currentChartType = 'line';
              if (this.isOnlineIndex) {
                this.queryNsdkRealTimeData();
              } else {
                queryOfflineRealtimeData(indexCode);
                queryOfflineChartData(indexCode);
              }
            } else {
              // 其他情况则切到1M、1个月范围、日，查线下的历史数据
              this.dateOptionValue = '1M';
              this.startAndEndTime = [ultls.getPreMonthDay(), ultls.getTodayValue()];
              this.chartFrequencyType = 'day';
              this.currentChartType = 'line';
              queryIndexHistoryData(indexCode);
            }
          }
        }
      },
      mounted: function () {
        this.isLoading = false;
        // 如果是国证恒生指数，走这里的逻辑 ------
        if (this.isHsiIndex) {
          // 恒生指数的日期区间默认切到1D
          this.dateOptionValue = '1D';
          // 恒生指数不是新版债券指数，默认有实时行情
          this.newIndexType = false;
          this.realtimemarket = '1';
          this.isRealtimeMarket = true;
          this.showDetail = '1';
          // 查询实时行情或历史行情数据
          this.handleHsiIndexPerformance(indexCode, 1);
          // 获取右侧资料下载地址
          this.queryHsiMaterialDownload();
          // 获取右侧指数简介、指数编制
          this.queryHsiIndexInfo(indexCode);
        } else {
          // 否则保持原逻辑 ------
          // 查询实时详情
          queryIndexConfigInfo(indexCode);
          // 判断是不是新版债券指数
          queryJudgeBondIndexType(indexCode);
          // 查询所有线上指数
          this.queryAllOnlineIndex();
          // 获取右侧资料下载地址
          this.colorPagePDF();
          this.colorPageBZFA();
          // 获取右侧指数简介、指数编制
          this.queryIndexInfo(indexCode);
          // 查询相关新闻
          this.queryNewsList(indexCode);
          // 设定资料下载默认链接地址
          if (window.isEN) {
            this.docDownload =
              '/en/module/pdf-detail.html?pdf=' + '/docs/gz_' + indexCode + '_e.pdf&name=' + this.indexName + '&indexCode=' + indexCode + '&type=1';
            this.colorPageDownload =
              '/en/module/pdf-detail-pro.html?pdf=' +
              '/docs/jj_' +
              indexCode +
              '_e.pdf&name=' +
              this.indexName +
              '&indexCode=' +
              indexCode +
              '&type=2';
          } else {
            this.docDownload =
              '/module/pdf-detail.html?pdf=' + '/docs/gz_' + indexCode + '.pdf&name=' + this.indexName + '&indexCode=' + indexCode + '&type=1';
            this.colorPageDownload =
              '/module/pdf-detail-pro.html?pdf=' + '/docs/jj_' + indexCode + '.pdf&name=' + this.indexName + '&indexCode=' + indexCode + '&type=2';
          }
        }
        // 公共逻辑 ------
        // 根据路由判断是否跳转到“样本详情”tab
        this.checkJumpActiveName(indexCode);
        // 获取调整分析指标
        this.getIndexAnalysis(indexCode);
        // 下拉加载函数
        window.addEventListener('scroll', this.getPageCountRequest);
        // 移动端初始化插件
        if (window.isMobile) {
          var calendar = new datePicker();
          calendar.init({
            trigger: '#mIndexDateStart' /*按钮选择器，用于触发弹出插件*/,
            type: 'date' /*模式：date日期；datetime日期时间；time时间；ym年月；*/,
            minDate: '1900-1-1' /*最小日期*/,
            maxDate: '2100-12-31' /*最大日期*/,
            onSubmit: function () {
              /*确认时触发事件*/
            },
            onClose: function () {
              /*取消时触发事件*/
            }
          });
          var newCalendar = new datePicker();
          newCalendar.init({
            trigger: '#mIndexDateEnd' /*按钮选择器，用于触发弹出插件*/,
            type: 'date' /*模式：date日期；datetime日期时间；time时间；ym年月；*/,
            minDate: '1900-1-1' /*最小日期*/,
            maxDate: '2100-12-31' /*最大日期*/,
            onSubmit: function () {
              /*确认时触发事件*/
            },
            onClose: function () {
              /*取消时触发事件*/
            }
          });

          var chartCalender = new datePicker();

          chartCalender.init({
            trigger: '#mChartDateStart' /*按钮选择器，用于触发弹出插件*/,
            type: 'date' /*模式：date日期；datetime日期时间；time时间；ym年月；*/,
            minDate: '1900-1-1' /*最小日期*/,
            maxDate: '2100-12-31' /*最大日期*/,
            onSubmit: function () {
              /*确认时触发事件*/
            },
            onClose: function () {
              /*取消时触发事件*/
            }
          });

          var newChartCalender = new datePicker();

          newChartCalender.init({
            trigger: '#mChartDateEnd' /*按钮选择器，用于触发弹出插件*/,
            type: 'date' /*模式：date日期；datetime日期时间；time时间；ym年月；*/,
            minDate: '1900-1-1' /*最小日期*/,
            maxDate: '2100-12-31' /*最大日期*/,
            onSubmit: function () {
              /*确认时触发事件*/
            },
            onClose: function () {
              /*取消时触发事件*/
            }
          });
        }
      },
      computed: {
        freqRange: function () {
          if (window.isMobile) {
            // 移动端暂无日期区间选择，暂时都默认返回month
            return 'month';
          } else {
            if (this.activeName >= 2) return 'month';
            var key = this.activeName == 0 ? 'startAndEndTime' : 'startAndEndTimeHistory';
            var dateOptionKey = this.activeName == 0 ? 'dateOptionValue' : 'dateOptionHistory';
            if (!this[key] || this[key].length === 0) return 'month';
            var range = 1 + (new Date(this[key][1]) - new Date(this[key][0])) / 1000 / 3600 / 24;
            var startMonth = new Date(this[key][0]).getMonth() + 1;
            var endMonth = new Date(this[key][1]).getMonth() + 1;
            var startYear = new Date(this[key][0]).getFullYear();
            var endYear = new Date(this[key][1]).getFullYear();
            if (this[dateOptionKey] === 'ALL' || (range >= 7 && (endMonth !== startMonth || endYear !== startYear))) {
              return 'month';
            } else if (range >= 7) {
              return 'week';
            } else {
              return 'day';
            }
          }
        },
        // 是否为国证恒生指数，是的话如下控件不会显示：
        // 指数表现tab中的：图表+频次筛选，指数对比功能，图形和表格
        // 历史行情中的：频次筛选
        // 样本详情中的：历史样本下载和历史调样下载
        isHsiIndex: function () {
          return hsiCodeList.includes(indexCode);
        },
        // 是否为国证恒生大湾区数字经济指数
        isSZHKDE: function () {
          return indexCode === hsiCodeList[0];
        },
        // 是否为国证恒生大湾区消费指数
        isSZHKCON: function () {
          return indexCode === hsiCodeList[1];
        }
      },
      methods: {
        // 国证恒生指数-指数表现-查询数据函数
        handleHsiIndexPerformance(code, flag) {
          // tab数据只渲染一次即可，轮询时无需再赋值
          if (flag === 1) {
            let cnTabList = [
              { name: '指数表现', id: '0' },
              { name: '历史行情', id: '1' },
              { name: '样本详情', id: '2' }
            ];
            let enTabList = [
              { name: 'Index Performance', id: '0' },
              { name: 'Historical Data', id: '1' },
              { name: 'Constituents', id: '2' }
            ];
            // PC端和移动端tab不展示"相关产品"的tab
            this.tabsList = window.isEN ? enTabList : cnTabList;
            this.mobileTabsList = this.mobileTabsList.filter(item => item.id !== '4');
            // 不展示表格数据
            this.showEchart = false;
          }
          // 如果当前选择1D，则读取实时行情接口，否则读取往期历史行情接口
          // 但是由于目前恒生指数默认切到“ALL”，没有实时行情，所以直接读历史行情，然后实时行情只取名字即可，所以下面这行注释等恢复实时行情后再解开
          this.dateOptionValue === '1D' ? this.queryHsiRealTimeData(code) : this.queryHsiHistoryData(code);
          // this.queryHsiIndexData(code);
          // this.queryHsiHistoryData(code);
        },
        // 国证恒生指数-指数表现-取恒生指数的指数信息
        queryHsiIndexData(code) {
          $.ajax({
            url: this.hsiIndexData.common.realtimeUrl,
            type: 'get',
            async: true,
            data: { code },
            dataType: 'json'
          }).done(res => {
            if (res.code === 200 && res.data) {
              const resData = res.data;
              this.indexName = window.isEN ? resData.indexEName : resData.indexName;
              this.indexCodeName = resData.indexCode;
              this.indexSource = resData.indexSource;
              this.indexType = resData.indexType;
            } else {
              this.indexName = '';
              this.indexCodeName = '';
            }
            // 更新移动端顶部信息。目前恒生指数没有实时行情，所以不用展示“交易时间”
            this.setDomText({ domId: 'mobIndexName', text: this.indexName });
          });
        },
        // 国证恒生指数-指数表现-查询实时行情数据
        queryHsiRealTimeData(code) {
          $.ajax({
            url: this.hsiIndexData.common.realtimeUrl,
            type: 'get',
            async: true,
            data: { code },
            dataType: 'json'
          }).done(res => {
            this.showStockChart = true;
            // 接口返回错误时，图形渲染空数据
            if (res.code !== 200) {
              createTimeChart([], 'line', 1, '');
              return;
            }
            // 接口正常时，渲染数据
            const resData = res.data || {};
            const arrData = resData.data || [];
            // 格式化数据
            let arrFormateData = arrData.map((arrItem, index) => {
              const diffValue = ultls.valueToNumber(arrItem[6]);
              const pctChgValue = ultls.valueToNumber(arrItem[7]);
              const date = arrItem[0] || '';
              // 后端返回的时间是年月日时分秒毫秒格式，需要格式化处理
              const dateTime = ultls.hsiFormatterDate(date);
              return {
                id: index,
                x: index,
                y: ultls.valueToNumber(arrItem[1]),
                xLabel: dateTime ? dateTime.substring(11, 16) : '', // 截取时分
                extra: {
                  date: dateTime ? dateTime.substring(0, 16) : '', // 不要秒
                  now: ultls.valueToNumber(arrItem[1]),
                  diff: diffValue,
                  diffLabel: `${diffValue <= 0 ? '' : '+'}${diffValue.toFixed(2)}`,
                  pctChg: pctChgValue,
                  pctChgLabel: `${pctChgValue <= 0 ? '' : '+'}${pctChgValue.toFixed(2)}%`
                }
              };
            });
            // 恒生指数详情取最后一个展示
            const lastData = arrFormateData[arrFormateData.length - 1];
            // 赋值当前行情信息
            this.indexName = window.isEN ? resData.indexEName : resData.indexName;
            this.indexCodeName = resData.indexCode;
            this.indexSource = resData.indexSource;
            this.indexType = resData.indexType;
            if (lastData) {
              this.indexCurrent = lastData.y.toFixed(2);
              this.indexChg = lastData.extra.diff.toFixed(2);
              this.indexPercent = lastData.extra.pctChg.toFixed(2);
              this.indexTime = ultls.formateDateTime(lastData.extra.date);
            } else {
              this.indexCurrent = '--';
              this.indexChg = '--';
              this.indexPercent = '--';
              this.indexTime = '--';
            }
            // 移动端顶部补一下数据
            this.setDomText({ domId: 'mobIndexName', text: this.indexName });
            this.setDomText({ domId: 'mobIndexTime', text: this.indexTime });
            // 如果点位数组不是空数据，那么需要补齐后面的数据到下午16:00，以让横坐标显示完整
            // 空数据就不用补了，直接会显示“没有数据”
            if (arrFormateData.length > 0) {
              this.fillHsiRealTimeData(arrFormateData);
            }
            // 渲染图形数据
            createHsiIndexChart(arrFormateData, 'realtime');
          });
        },
        // 国证恒生指数-指数表现-判断是否要补充横坐标点位数据
        fillHsiRealTimeData(arrData) {
          // 取最后一个的时间戳进行判断，看是否需要补足
          const lastData = arrData[arrData.length - 1];
          const lastDate = lastData.extra.date;
          const minuteTime = 60000; // 一分钟对应的毫秒
          const lastTime = new Date(lastDate).getTime(); // 最后一个时间戳
          const theDate = ultls.formateDate(lastTime); // 当天的年-月-日
          const amEndTime = new Date(`${theDate} 12:00:00`).getTime(); // 上午收盘时间戳
          const pmStartTime = new Date(`${theDate} 13:00:00`).getTime(); // 下午开盘时间戳
          const pmEndTime = new Date(`${theDate} 16:00:00`).getTime(); // 下午收盘时间戳
          const wholePmNum = (pmEndTime - pmStartTime) / minuteTime; // 整个下午有几个点位（一分钟一个点位）
          let lastId = lastData.id; // 最后一个数据的id，用来给下面补的数据做起始id值，然后逐渐+1
          let xLabelTime = ''; // 每次循环时，当前点位的时间戳
          // 如果是在上午12:00之前，先补上午、再补下午
          if (lastTime < amEndTime) {
            const amDiffNum = (amEndTime - lastTime) / minuteTime; // 计算相差几个点位
            for (let i = 0; i < amDiffNum; i++) {
              lastId += 1; // id递增+1
              xLabelTime = lastTime + (i + 1) * minuteTime; // 当前点位的时间戳
              // 补的数据，除了id、x和横坐标日期外，都是空值
              arrData.push({
                id: lastId,
                x: lastId,
                y: null,
                xLabel: ultls.formateDateTime(xLabelTime).substring(11, 16), // 取年-月-日 时:分
                extra: {
                  date: ultls.formateDateTime(xLabelTime).substring(0, 16), // 取时:分
                  diff: '',
                  diffLabel: '',
                  now: '',
                  pctChg: '',
                  pctChgLabel: ''
                }
              });
            }
            // 补整个下午
            for (let n = 0; n <= wholePmNum; n++) {
              lastId += 1;
              xLabelTime = pmStartTime + n * minuteTime;
              arrData.push({
                id: lastId,
                x: lastId,
                y: null,
                xLabel: ultls.formateDateTime(xLabelTime).substring(11, 16),
                extra: {
                  date: ultls.formateDateTime(xLabelTime).substring(0, 16),
                  diff: '',
                  diffLabel: '',
                  now: '',
                  pctChg: '',
                  pctChgLabel: ''
                }
              });
            }
          } else if (lastTime === amEndTime) {
            // 如果是中午休盘时间，补足下午
            for (let j = 0; j <= wholePmNum; j++) {
              lastId += 1;
              xLabelTime = pmStartTime + j * minuteTime;
              arrData.push({
                id: lastId,
                x: lastId,
                y: null,
                xLabel: ultls.formateDateTime(xLabelTime).substring(11, 16),
                extra: {
                  date: ultls.formateDateTime(xLabelTime).substring(0, 16),
                  diff: '',
                  diffLabel: '',
                  now: '',
                  pctChg: '',
                  pctChgLabel: ''
                }
              });
            }
          } else if (lastTime < pmEndTime) {
            // 下午时间，在收盘时间内
            const pmDiffNum = (pmEndTime - lastTime) / minuteTime;
            for (let k = 0; k < pmDiffNum; k++) {
              lastId += 1;
              xLabelTime = lastTime + (k + 1) * minuteTime;
              arrData.push({
                id: lastId,
                x: lastId,
                y: null,
                xLabel: ultls.formateDateTime(xLabelTime).substring(11, 16),
                extra: {
                  date: ultls.formateDateTime(xLabelTime).substring(0, 16),
                  diff: '',
                  diffLabel: '',
                  now: '',
                  pctChg: '',
                  pctChgLabel: ''
                }
              });
            }
          }
          // 下午时间，超出收盘时间的，就不处理了
        },
        // 国证恒生指数-指数表现-查询历史行情数据
        queryHsiHistoryData(code) {
          $.ajax({
            url: this.hsiIndexData.common.historyUrl,
            type: 'get',
            data: {
              indexCode: code,
              start: this.startAndEndTime[0] || '',
              end: this.startAndEndTime[1] || ''
            },
            dataType: 'json'
          }).done(res => {
            this.showStockChart = true;
            // 接口错误或者无数据时展示空
            if (res.code !== 200 || !Array.isArray(res.data) || res.data.length === 0) {
              createHsiIndexChart([], 'history');
              return;
            }
            // 数据格式化，匹配图表
            const arrData = res.data.reverse();
            const arrFormateData = arrData.map((item, index) => {
              // 计算涨跌幅（目前先不展示涨跌幅，因为接口没有返回。暂时不展示自己计算的，因为会有误差）
              const diff = Number(item.prior) && Number(item.close) ? Number(item.close) - Number(item.prior) : 0;
              const diffValue = ultls.valueToNumber(diff);
              const pctChgValue = ultls.valueToNumber(item.pctChg);
              const date = ultls.formateDate(item.tradeDate);
              return {
                id: index,
                x: index,
                y: ultls.valueToNumber(item.close),
                xLabel: date,
                extra: {
                  date,
                  open: ultls.valueToNumber(item.prior),
                  high: ultls.valueToNumber(item.high),
                  low: ultls.valueToNumber(item.low),
                  close: ultls.valueToNumber(item.close),
                  diff: diffValue,
                  diffLabel: `${diffValue <= 0 ? '' : '+'}${diffValue.toFixed(2)}`,
                  pctChg: pctChgValue,
                  pctChgLabel: `${pctChgValue <= 0 ? '' : '+'}${pctChgValue.toFixed(2)}%`
                }
              };
            });
            // 渲染图表
            createHsiIndexChart(arrFormateData, 'history');
          });
        },
        // 国证恒生指数-写入右侧资料下载地址
        queryHsiMaterialDownload() {
          const indexData = this.hsiIndexData[indexCode];
          if (window.isEN) {
            this.docDownloadPDF = indexData.enMaterialUrl.scheme;
            this.colorPageDownloadPDF = indexData.enMaterialUrl.colorPage;
          } else {
            this.docDownloadPDF = indexData.cnMaterialUrl.scheme;
            this.colorPageDownloadPDF = indexData.cnMaterialUrl.colorPage;
          }
          this.$forceUpdate();
        },
        // 国证恒生指数-写入右侧指数简介、指数编制
        queryHsiIndexInfo(code) {
          const indexData = this.hsiIndexData[code];
          this.indexInfo = window.isEN ? indexData.enIndexInfo : indexData.cnIndexInfo;
        },
        // 国证恒生指数-历史行情-查询列表数据
        queryHsiHistoryList(vueTemp) {
          $.ajax({
            url: vueTemp.hsiIndexData.common.historyUrl,
            type: 'get',
            data: {
              indexCode: indexCode,
              start: vueTemp.startAndEndTimeHistory[0] || '',
              end: vueTemp.startAndEndTimeHistory[1] || ''
            },
            dataType: 'json'
          }).done(function (res) {
            vueTemp.historyLoading = false;
            if (res.code !== 200) {
              return;
            }
            // 数据格式化，匹配原逻辑
            const arrData = res.data || [];
            let arrList = arrData.map(item => {
              return {
                date: ultls.formateDate(item.tradeDate),
                open: item.prior,
                high: item.high,
                low: item.low,
                close: item.close,
                percent: item.pctChg ? `${item.pctChg}%` : '',
                amount: 0,
                volume: 0
              };
            });
            vueTemp.historyPage = 1;
            vueTemp.historyAllData = arrList;
            vueTemp.historyAllDataTotal = res.total || 0;
            vueTemp.historyShowData = arrList.length < 20 ? arrList : arrList.slice(0, 20);
          });
        },
        // 国证恒生指数-历史行情-下载
        hsiHistoryDownload() {
          const options = {
            url: this.hsiIndexData.common.historyDownUrl,
            method: 'get',
            params: {
              indexCode,
              start: this.startAndEndTimeHistory[0] || '',
              end: this.startAndEndTimeHistory[1] || ''
            }
          };
          ultls.download(options);
        },
        // 国证恒生指数-样本详情-查询列表数据
        queryHsiSampleList(pageNo, pageSize, vueTemp) {
          $.ajax({
            url: vueTemp.hsiIndexData.common.sampleUrl,
            type: 'get',
            data: {
              indexCode,
              // 英文版的移动端用的是sampleDate字段，其他用的是currDate
              month: window.isEN && window.isMobile ? vueTemp.sampleDate : vueTemp.currDate,
              pageNum: pageNo,
              rows: pageSize
            },
            dataType: 'json'
          }).done(function (res) {
            vueTemp.sampleLoading = false;
            if (res.code !== 200 || !res.data) {
              return;
            }
            const arrData = res.data.rows || [];
            const arrList = arrData.map(item => {
              return {
                dateStr: ultls.formateDate(item.tradate),
                seccode: item.stkCode,
                secname: window.isEN ? item.stkNameEn : item.stkName,
                trade: window.isEN ? item.industryEn : item.industry,
                weight: item.pctWgt ? Number(item.pctWgt).toFixed(2) : ''
              };
            });
            vueTemp.sampleList = pageNo === 1 ? arrList : [...vueTemp.sampleList, ...arrList];
            vueTemp.sampleTotal = res.data.total || 0;
          });
        },
        // 国证恒生指数-样本详情-下载
        hsiSampleDownload(vueTemp) {
          const options = {
            url: vueTemp.hsiIndexData.common.sampleDownUrl,
            method: 'get',
            params: { indexCode, month: vueTemp.currDate }
          };
          ultls.download(options);
        },
        // 根据路由判断是否跳转到“样本详情”tab
        checkJumpActiveName(indexCode) {
          const search = window.location.search;
          let redirectedToSampleDetail = false;
          if (search) {
            var arr = search.substr(1, search.length - 1).split('&');
            for (var i = 0; i < arr.length; i++) {
              var e = arr[i].split('=');
              if (e[0] === 'activeName' && e[1] === '2') {
                this.activeName = '2';
                this.mainTabClick({ name: 2 });
                redirectedToSampleDetail = true;
                break;
              }
            }
          }
          // 如果是国证恒生指数 或者 直接跳到样本详情tab的话，就不走下面的逻辑
          if (this.isHsiIndex || redirectedToSampleDetail) {
            return;
          }
          // 否则，请求市场分布、行业占比数据
          if (!window.isMobile) {
            this.queryMarketDistribute(indexCode);
          }
          // 请求阶段性收益
          this.queryIncomeInfo(indexCode);
        },
        colorPageBZFA: function () {
          var self = this;
          var bzfaUrl = '/info/getColorPageBZFA?indexCode=' + indexCode;
          $.ajax({
            url: bzfaUrl,
            type: 'get',
            dataType: 'json'
          }).done(function (res) {
            if (res.code == 200 && res.data) {
              if (window.isEN) {
                self.docDownloadPDF = '/en/module/pdf-detail.html?pdf=' + '/docs/gz_' + res.data + '&indexCode=' + indexCode + '&type=1';
              } else {
                self.docDownloadPDF = '/module/pdf-detail.html?pdf=' + '/docs/gz_' + res.data + '&indexCode=' + indexCode + '&type=1';
              }
              self.$forceUpdate();
            }
          });
        },
        colorPagePDF: function () {
          if (indexCode === '932604') {
            var self = this;
            const indexData = this[indexCode];
            if (window.isEN) {
              // this.docDownloadPDF = indexData.enMaterialUrl.scheme;
              self.colorPageDownloadPDF = indexData.enMaterialUrl.colorPage;
            } else {
              // this.docDownloadPDF = indexData.cnMaterialUrl.scheme;
              self.colorPageDownloadPDF = indexData.cnMaterialUrl.colorPage;
            }
            this.$forceUpdate();
            return;
          }
          var pdfUrl = '/info/getPdfPath?indexCode=' + indexCode;
          $.ajax({
            url: pdfUrl,
            type: 'get',
            dataType: 'json'
          }).done(function (res) {
            if (res.code == 200 && res.data) {
              if (window.isEN) {
                self.colorPageDownloadPDF = '/en/module/pdf-detail-pro.html?pdf=' + res.data + '&indexCode=' + indexCode + '&type=2';
              } else {
                self.colorPageDownloadPDF = '/module/pdf-detail-pro.html?pdf=' + res.data + '&indexCode=' + indexCode + '&type=2';
              }
              self.$forceUpdate();
            }
          });
        },
        // PC端切换tab
        mainTabClick: function (value) {
          const startAndEndTimeHistory = this.startAndEndTimeHistory || [];
          // 切换历史行情
          if (value.name == 1) {
            // 有数据就不做处理，无数据就再请求一次
            if (this.historyAllData.length > 0) return;
            // 当时间范围没值，但又选中了1个月时，将时间范围补成一个月区间
            if (startAndEndTimeHistory.length < 2 && this.dateOptionHistory === '1M') {
              this.startAndEndTimeHistory = [ultls.getPreMonthDay(), ultls.getTodayValue()];
            }
            this.queryIndexHistoryList();
          }
          // 切换样本详情
          if (value.name == 2) {
            // 有数据就不做处理，无数据就再请求一次
            if (this.sampleList.length > 0) return;
            this.querySampleList(1, 20, '1');
          }
          // 切换相关产品
          if (value.name == 3) {
            if (this.fundList.length > 0) return;
            this.queryFundList(1, 20);
          }
        },
        // 指数表现tab-切换日期区间
        changeDateOption: function (value) {
          // 非指数对比时
          if (!this.isContrast) {
            this.dateOptionValue = value;
            const isMinuteFrequency = ['m1', 'm15', 'm30', 'm60'].includes(this.chartFrequencyType);
            // 选择1D时
            if (value === '1D') {
              // 1D没有K线图，所以要切回折线图；频次切回m1；时间范围切回1天
              if (this.currentChartType === 'candlestick') {
                this.currentChartType = 'line';
              }
              this.chartFrequencyType = 'm1';
              this.startAndEndTime = [ultls.getTodayValue(), ultls.getTodayValue()];
              // 恒生指数走独立逻辑，更新实时行情和分时图
              if (this.isHsiIndex) {
                this.queryHsiRealTimeData(indexCode);
              } else if (this.isOnlineIndex) {
                // 线上指数请求NSDK更新实时行情和分时图
                this.queryNsdkRealTimeData();
              } else {
                // 其他情况：既然能出现1D选项，那必然是线下有实时行情的指数，那就查数据库接口更新实时行情和分时图
                queryOfflineRealtimeData(indexCode);
                queryOfflineChartData(indexCode);
              }
            } else if (value == '1M') {
              // 选1M时，没有分钟频次，要切回day；时间范围切成1个月
              if (isMinuteFrequency) {
                this.chartFrequencyType = 'day';
              }
              this.startAndEndTime = [ultls.getPreMonthDay(), ultls.getTodayValue()];
              // 恒生指数走独立逻辑，更新历史图表数据
              if (this.isHsiIndex) {
                this.queryHsiHistoryData(indexCode);
              } else if (this.isOnlineIndex && this.chartFrequencyType === 'day') {
                // 线上指数、频次是day，则请求NSDK获取历史图表数据
                this.queryNsdkKlineHistoryData();
              } else {
                // 其他情况都查数据库更新历史图表数据
                queryIndexHistoryData(indexCode);
              }
            } else if (value == '6M') {
              // 选6M时，没有分钟频次，要切回day；时间范围切成6个月
              if (isMinuteFrequency) {
                this.chartFrequencyType = 'day';
              }
              this.startAndEndTime = [ultls.getSixMonthDay(), ultls.getTodayValue()];
              // 恒生指数走独立逻辑，更新历史图表数据
              if (this.isHsiIndex) {
                this.queryHsiHistoryData(indexCode);
              } else if (this.isOnlineIndex && this.chartFrequencyType === 'day') {
                // 线上指数、频次是day，则请求NSDK获取历史图表数据
                this.queryNsdkKlineHistoryData();
              } else {
                // 其他情况都查数据库更新历史图表数据
                queryIndexHistoryData(indexCode);
              }
            } else if (value == '1Y') {
              // 选6M时，没有分钟频次，要切回day；时间范围切成1年
              if (isMinuteFrequency) {
                this.chartFrequencyType = 'day';
              }
              this.startAndEndTime = [ultls.getPreYearDay(), ultls.getTodayValue()];
              // 国证恒生指数
              if (this.isHsiIndex) {
                this.queryHsiHistoryData(indexCode);
              } else if (this.isOnlineIndex && this.chartFrequencyType === 'day') {
                // 线上指数且频次是“日”
                this.queryNsdkKlineHistoryData();
              } else {
                queryIndexHistoryData(indexCode);
              }
            } else {
              // 剩下是选择"ALL"的情况，全球指数频次切换为day，其他的切换为week
              if (this.indexType == 700) {
                this.chartFrequencyType = 'day';
              } else {
                this.chartFrequencyType = 'week';
              }
              this.startAndEndTime = [];
              // 国证恒生指数
              if (this.isHsiIndex) {
                this.queryHsiHistoryData(indexCode);
              } else if (this.isOnlineIndex && this.chartFrequencyType === 'day') {
                // 线上指数且频次是“日”，日期采用基日开始时间和最大结束时间
                this.queryNsdkKlineHistoryData();
              } else {
                // 其他情况
                queryIndexHistoryData(indexCode);
              }
            }
          }
          // 指数对比时
          else {
            this.dateOptionValue = value;
            if (value == '1D') {
              //移除多余收益率列表
              this.incomeList = this.incomeList.slice(0, 1);
            }
            if (this.contrastIndexList.length > 0) {
              var i = 0,
                str = indexCode + ';';
              for (i; i < this.contrastIndexList.length; i++) {
                str += this.contrastIndexList[i].indexCode + ';';
              }
              // 切到1D时，要关闭指数对比，请求分时数据
              if (value == '1D') {
                this.contrastIndexList = [];
                this.chartFrequencyType = 'm1';
                this.startAndEndTime = [];
                this.dateOptionValue = '1D';
                if (this.isOnlineIndex) {
                  this.queryNsdkRealTimeData();
                } else {
                  queryOfflineRealtimeData(indexCode);
                  queryOfflineChartData(indexCode);
                }
                return;
              } else if (value == '1M') {
                vm.startAndEndTime = [ultls.getPreMonthDay(), ultls.getTodayValue()];
              } else if (value == '6M') {
                this.chartFrequencyType = 'day';
                vm.startAndEndTime = [ultls.getSixMonthDay(), ultls.getTodayValue()];
              } else if (value == '1Y') {
                this.chartFrequencyType = 'day';
                vm.startAndEndTime = [ultls.getPreYearDay(), ultls.getTodayValue()];
              } else {
                this.chartFrequencyType = 'day';
                vm.startAndEndTime = [];
              }
              queryIndexContrastData(str);
              this.isContrast = true;
            } else {
              var str = indexCode + ';';
              queryIndexContrastData(str);
              this.isContrast = false;
            }
          }
        },
        // 指数表现tab-切换时间范围
        changeChartDate: async function () {
          // 时间范围插件有bug，点击x按钮清空时会报错并且清空不掉，这里延迟一下
          await this.promiseSomeTime(0);
          // 统一处理PC和移动端的时间范围判断
          // 时间范围为空时（不用长度判断是因为移动端可能会造成[empty， empty]这种情况）
          if (!this.startAndEndTime || (!this.startAndEndTime[0] && !this.startAndEndTime[1])) {
            this.startAndEndTime = [];
          } else if (!this.startAndEndTime[0]) {
            // 只选了一个值时，提示另一个值未选（移动端才会出现，以前接口支持1个值，现在加了NSDK，必须要2个值）
            this.$message.error('请选择开始日期！');
            return;
          } else if (!this.startAndEndTime[1]) {
            this.$message.error('请选择结束日期！');
            return;
          } else if (Date.parse(this.startAndEndTime[0]) > Date.parse(this.startAndEndTime[1])) {
            this.$message.error('开始日期不能大于结束日期！');
            return;
          }
          this.mobileFullScreenDialog = false;
          // 切换时间范围后，清空日期区间值，频次切为day
          this.dateOptionValue = '';
          this.chartFrequencyType = 'day';
          // 指数对比时
          if (this.isContrast) {
            if (this.contrastIndexList.length > 0) {
              var i = 0,
                str = indexCode + ';';
              for (i; i < this.contrastIndexList.length; i++) {
                str += this.contrastIndexList[i].indexCode + ';';
              }
              queryIndexContrastData(str);
              this.isContrast = true;
            } else {
              var str = indexCode + ';';
              queryIndexContrastData(str);
              this.isContrast = false;
            }
            return;
          }
          // 非指数对比时
          // 国证恒生指数走独立逻辑更新历史图表数据
          if (this.isHsiIndex) {
            this.queryHsiHistoryData(indexCode);
            return;
          }
          // 如果日期范围被清空，频次切换到week，请求数据库更新历史图表数据
          if (!this.startAndEndTime || this.startAndEndTime.length === 0) {
            this.chartFrequencyType = 'week';
            queryIndexHistoryData(indexCode);
            return;
          }
          // 日期范围没有清空而是变更，那线上指数请求NSDK更新历史图表数据
          if (this.isOnlineIndex) {
            this.queryNsdkKlineHistoryData();
            return;
          }
          // 其他情况请求数据库更新历史图表数据
          queryIndexHistoryData(indexCode);
        },
        // 指数表现tab-切换图表类型
        changeChartType: function (val) {
          // 切换图表类型不需要重新请求，利用缓存进行图表切换
          this.currentChartType = val;
          // 面积图
          if (val === 'area') {
            if (this.dateOptionValue == '1D') {
              createTimeChart(timeData, 'area', this.chartFrequencyType, preClosePrice);
            } else {
              createHistoryChart(newHistoryData, 'area');
            }
          } else if (val === 'line') {
            // 折线图
            if (this.dateOptionValue == '1D') {
              createTimeChart(timeData, 'line', this.chartFrequencyType, preClosePrice);
            } else {
              createHistoryChart(newHistoryData, 'line');
            }
          } else {
            // K线图
            createKlineChart(ohclData);
          }
        },
        // 指数表现tab-切换频次
        clickChartFrequency: function (val) {
          if (this.chartFrequencyType == val) {
            return;
          }
          this.chartFrequencyType = val;
          const currentChartType = this.currentChartType;
          // 如果是在分钟频次之间切换，那么当前日期区间一定是1D，要更新分时图，此时用缓存更新分时图即可
          if (['m1', 'm15', 'm30', 'm60'].includes(val)) {
            createTimeChart(timeData, currentChartType, val, preClosePrice);
            return;
          }
          // 剩下的则是更新历史图表数据
          // 如果频次为day且是线上指数且时，请求NSDK更新
          if (val === 'day' && this.isOnlineIndex) {
            this.queryNsdkKlineHistoryData();
            return;
          }
          // 其他情况请求数据库更新
          queryIndexHistoryData(indexCode);
        },
        // 指数表现tab-每5分钟更新一次实时行情
        setUpdateEveryFiveMinute() {
          setInterval(() => {
            if (vm.dateOptionValue === '1D') {
              // 线上指数请求NSDK更新
              if (vm.isOnlineIndex) {
                vm.queryNsdkRealTimeData();
              } else {
                // 线下指数有实时行情的请求接口更新
                queryOfflineRealtimeData(indexCode);
                queryOfflineChartData(indexCode);
              }
            }
          }, 5 * 60 * 1000);
        },
        // 指数表现tab-NSDK查询行情数据和分时图数据
        async queryNsdkRealTimeData() {
          // 业务期望交易日9.00-9.30之间也要展示数据，但NSDK在这期间清盘无数据，所以交易日的这半小时要查询上一个交易日的数据来展示
          let params = { indexCode, arrField: ['Time', 'PrePrice', 'Price', 'Volume', 'Amount'] };
          let requestResult = null;
          const requestType = await this.checkNsdkRequestType();
          if (requestType === 'realtimeData') {
            requestResult = await NsdkRequest.SDK_REQUEST_TREND(params).catch(err => err);
          } else {
            params.tradeDate = this.indexTradeInfo.lastTradeDay;
            requestResult = await NsdkRequest.SDK_REQUEST_HISTREND(params).catch(err => err);
          }
          if (requestResult.code !== 200) {
            return;
          }
          const allArrData = requestResult.data || [];
          const arrTime = allArrData[0] || [];
          const arrPrePrice = allArrData[1];
          const arrPrice = allArrData[2];
          const arrVolume = allArrData[3];
          const arrAmount = allArrData[4];
          let totalAmount = 0;
          let totalVolume = 0;
          // 用时间数组去遍历，组合数据格式
          const arrChartData = arrTime.map((_item, index) => {
            totalAmount += arrAmount[index];
            totalVolume += arrVolume[index];
            const objItem = {
              x: arrTime[index],
              y: arrPrice[index],
              extra: [
                arrTime[index], // 时间戳
                arrPrice[index], // 当前点位
                null, // 最高价
                arrPrice[0], // 开盘价
                null, // 最低价
                null, // 收盘价
                arrPrice[index] - arrPrePrice[0], // 涨跌幅
                (arrPrice[index] - arrPrePrice[0]) / arrPrePrice[0], // 涨跌幅百分比
                totalAmount, // 总成交额
                totalVolume, // 总成交量
                arrPrePrice[0] // 均价
              ]
            };
            return objItem;
          });
          // 赋值行情数据
          const lastTimeData = arrChartData.length > 0 ? arrChartData[arrChartData.length - 1] : null;
          if (lastTimeData) {
            const arrLastExtra = lastTimeData.extra;
            vm.indexCurrent = arrLastExtra[1].toFixed(2);
            vm.indexChg = arrLastExtra[6].toFixed(2);
            vm.indexPercent = (arrLastExtra[7] * 100).toFixed(2);
            vm.indexTime = ultls.formateDateTime(arrLastExtra[0]);
            vm.indexMoney = ultls.moneyUnits(totalAmount) || '--';
          } else {
            vm.indexCurrent = '--';
            vm.indexChg = '--';
            vm.indexPercent = '';
            vm.indexTime = '--';
            vm.indexMoney = '--';
          }
          // 更新移动端顶部的交易时间
          vm.setDomText({ domId: 'mobIndexTime', text: vm.indexTime });
          // 绘制分时图。赋值timeData进行缓存，更新昨日收盘价
          if (arrChartData.length === 0) {
            timeData = [];
            preClosePrice = '';
          } else {
            // 如果是历史分时图，数据不用处理，如果是实时分时图，NSDK给的数据是截止到当前分钟的，所以我们需要补齐到下午收盘时间，用空值填充，然后再处理不同的频次
            timeData = requestType === 'realtimeData' ? this.fillNsdkTimeData(arrChartData) : arrChartData;
            preClosePrice = arrPrePrice[0];
          }
          createTimeChart(timeData, this.currentChartType, this.chartFrequencyType, preClosePrice);
        },
        // 针对线上指数，判断当前是该调用实时分时还是调用历史分时
        async checkNsdkRequestType() {
          const isAfterNight = ultls.checkAfterTheTime('09:30:00');
          const { isTradeDay } = await this.getIndexTradeInfo(indexCode);
          // 当前是交易日且9点后就读实时行情，否则读历史分时行情
          return isTradeDay && isAfterNight ? 'realtimeData' : 'historyData';
        },
        // 查询指数当前的交易日信息
        async getIndexTradeInfo(code) {
          return new Promise(resolve => {
            // 9:30之前并且已经查询过，就不用再查询，直接返回查询结果，9:36之后并且已经查询过，也不用再查询
            // 唯独30分-35分，要重新查询，是因为接口返回的isTradeDay字段在9:30会进行更新，而我的定时器是5分钟更新一遍
            const ifBefore930 = !ultls.checkAfterTheTime('09:30:00');
            const isAfter936 = ultls.checkAfterTheTime('09:36:00');
            if (this.indexTradeInfo.isDone && (ifBefore930 || isAfter936)) {
              resolve(this.indexTradeInfo);
              return;
            }
            // 没查过就接口查询
            $.ajax({
              url: '/market/market/getMarketDay',
              type: 'get',
              async: true,
              dataType: 'json',
              data: { codesValue: code }
            }).done(res => {
              // 有数据就取数据
              if (res.code === 200 && Array.isArray(res.data) && res.data.length > 0) {
                const firstData = res.data[0];
                Object.assign(this.indexTradeInfo, { isDone: true, isTradeDay: firstData.isMarketDy === '1', lastTradeDay: firstData.lastMarketDay });
              } else {
                // 没数据就视为非交易日，上一个交易日取昨天
                const nowTime = new Date().getTime();
                const yestoday = ultls.formateDate(nowTime - 24 * 3600 * 1000);
                Object.assign(this.indexTradeInfo, { isDone: true, isTradeDay: false, lastTradeDay: yestoday });
              }
              resolve(this.indexTradeInfo);
            });
          });
        },
        // NSDK给的数据是截止到当前分钟的，所以我们需要补齐到下午收盘时间，用空值填充，然后再处理不同的频次
        fillNsdkTimeData(arrData) {
          // 本身是空数据就直接返回空数据
          if (arrData.length === 0) {
            return arrData;
          }
          // 取最后一个的时间戳进行判断，看是否需要补足
          let arrWholeData = [...arrData];
          const minuteTime = 60000; // 一分钟对应的毫秒
          const ifMoreTimeIndex = this.moreTimeIndexList.find(item => item.indexCode === indexCode); // 有的指数是下午15:30收盘
          const lastTime = arrData[arrData.length - 1].x; // NSDK给到的最后一个时间戳
          const theDate = ultls.formateDate(lastTime); // 当天的年-月-日
          const amEndTime = new Date(`${theDate} 11:30:00`).getTime(); // 上午收盘时间戳
          const pmStartTime = new Date(`${theDate} 13:00:00`).getTime(); // 下午开盘时间戳
          const pmEndTime = ifMoreTimeIndex ? new Date(`${theDate} 15:30:00`).getTime() : new Date(`${theDate} 15:00:00`).getTime(); // 下午收盘时间戳
          const wholePmNum = (pmEndTime - pmStartTime) / minuteTime; // 整个下午有几个点位（一分钟一个点位）
          // 如果是在上午11:30之前，先上午、再补下午
          if (lastTime < amEndTime) {
            const amDiffNum = (amEndTime - lastTime) / minuteTime;
            for (let i = 0; i < amDiffNum; i++) {
              arrWholeData.push({
                x: lastTime + (i + 1) * minuteTime,
                y: null,
                extra: [lastTime + (i + 1) * minuteTime, null, null, null, null, null, null, null, null, null, null]
              });
            }
            for (let n = 0; n < wholePmNum; n++) {
              arrWholeData.push({
                x: pmStartTime + n * minuteTime,
                y: null,
                extra: [pmStartTime + n * minuteTime, null, null, null, null, null, null, null, null, null, null]
              });
            }
          } else if (lastTime === amEndTime) {
            // 如果是中午休盘时间，补足下午
            for (let j = 0; j < wholePmNum; j++) {
              arrWholeData.push({
                x: pmStartTime + j * minuteTime,
                y: null,
                extra: [pmStartTime + j * minuteTime, null, null, null, null, null, null, null, null, null, null]
              });
            }
          } else if (lastTime < pmEndTime) {
            // 下午时间，在收盘时间内
            const pmDiffNum = (pmEndTime - lastTime) / minuteTime;
            for (let k = 0; k < pmDiffNum; k++) {
              arrWholeData.push({
                x: lastTime + (k + 1) * minuteTime,
                y: null,
                extra: [lastTime + (k + 1) * minuteTime, null, null, null, null, null, null, null, null, null, null]
              });
            }
          }
          // 下午时间，超出收盘时间的，就不处理了
          // 再根据频次
          return arrWholeData;
        },
        // 指数表现tab-NSDK查询日K线图数据
        async queryNsdkKlineHistoryData() {
          const startAndEndTime = this.startAndEndTime || [];
          const params = {
            indexCode,
            arrField: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'PreClose'],
            dateRange: startAndEndTime.length > 0 ? startAndEndTime : [this.jrStartDate, this.jrEndDate],
            period: 'day'
          };
          const { code, data, response } = await NsdkRequest.SDK_REQUEST_KLINE(params).catch(err => err);
          if (code !== 200) {
            return;
          }
          this.klineDoDataResponse(response);
        },
        // 历史行情数据回调
        klineDoDataResponse: function (response) {
          const dataSet = response.data;
          const getDataSource = dataSet.getDataSource() || {};
          const allArrData = getDataSource.data || [];
          //表头数组
          let colInfos = dataSet.getColumnInfos();
          //获取行数
          let rowCount = dataSet.getRowCount();
          //更多获取数据方式，参考GDataSet提供的函数
          //获取列名
          let theadData = colInfos.map(info => {
            return info.getName();
          });
          let tbodyData = [];
          newHistoryData = [];
          ohclData = [];
          volumeData = [];
          for (let i = 0; i < rowCount; i++) {
            //根据行号获取行数数据
            let data = dataSet.getRowValues(i);
            //日期时间转时间戳
            data[0] = new Date(data[0]).getTime();
            tbodyData.push({ rowData: data });
            // 昨日收盘价和今日收盘价算出涨跌幅百分比
            const chg = Number(data[4] - data[7]);
            const percent = data[7] ? chg / data[7] : 0;
            var obj = {
              x: data[0],
              y: Number(data[4]),
              extra: [
                data[0],
                null,
                Number(data[2]),
                Number(data[1]),
                Number(data[3]),
                Number(data[4]),
                chg,
                percent,
                Number(data[6]),
                Number(data[5]),
                null
              ]
            };
            newHistoryData.push(obj);
            ohclData.push({
              x: data[0], //the date
              open: Number(data[1]), //open
              high: Number(data[2]), //high
              low: Number(data[3]), //low
              close: Number(data[4]), //close
              extra: [
                data[0],
                null,
                Number(data[2]),
                Number(data[1]),
                Number(data[3]),
                Number(data[4]),
                chg,
                percent,
                Number(data[6]),
                Number(data[5]),
                null
              ]
            });
            // 底部红绿柱
            volumeData.push({
              x: data[0], // 日期
              y: Number(data[5]), // 成交量
              color: chg < 0 ? 'green' : 'red'
            });
          }
          // 绘图
          vm.showStockChart = true;
          if (vm.currentChartType == 'line' || vm.currentChartType == 'area') {
            createHistoryChart(newHistoryData, vm.currentChartType);
          } else {
            createKlineChart(ohclData);
          }
        },
        // 历史行情 时间控件日期切换
        changeHistoryDate: async function (value) {
          // 时间范围框有问题，点击x清空日期时value值会被设置成null而非[]，然后导致插件报错，这里做一下特殊延迟处理
          if (!value) {
            await this.promiseSomeTime(0);
            this.startAndEndTimeHistory = [];
          } else {
            this.startAndEndTimeHistory = value;
          }
          this.dateOptionValue = '';
          this.queryIndexHistoryList();
        },
        // 请求市场分布、行业占比数据
        queryMarketDistribute: function (code) {
          var self = this;
          //市场分布
          $.ajax({
            url: window.hqUrl + '/market/market-distribute/market',
            type: 'get',
            data: {
              indexcode: code,
              lang: getCookie('language') ? getCookie('language') : 'zh_CN'
            },
            dataType: 'json'
          }).done(function (res) {
            if (res.code == 200) {
              var dataList = [],
                dataList2 = [],
                i = 0;

              if (res.data && res.data.length > 0) {
                for (i; i < res.data.length; i++) {
                  var obj = [res.data[i].name, res.data[i].weight.toFixed(2), res.data[i].companyNum],
                    obj1 = {
                      code: res.data[i].code,
                      name: res.data[i].name,
                      value: res.data[i].weight
                    };
                  dataList.push(obj);
                  dataList2.push(obj1);
                }
                //创建Echarts图表
                //环形嵌套图
                self.marketDate = res.message;

                if (window.isMobile) {
                  echartsPlugin.createMobileHoopChart('mecharts1', dataList, dataList2);
                } else {
                  echartsPlugin.createHoopChart('echarts1', dataList, dataList2);
                }
              } else {
                self.showEchart = false;
              }
            }
          });
          //行业占比
          $.ajax({
            url: window.hqUrl + '/market/market-distribute/trade',
            type: 'get',
            data: {
              indexcode: code,
              lang: getCookie('language') ? getCookie('language') : 'zh_CN'
            },
            dataType: 'json'
          }).done(function (res) {
            if (res.code == 200 && res.data) {
              var dataList = [],
                i = 0;
              for (i; i < res.data.length; i++) {
                var obj = {
                  code: res.data[i].code,
                  name: res.data[i].trade,
                  value: res.data[i].weight.toFixed(2)
                };
                dataList.push(obj);
              }
              self.tradeDate = res.message;
              //创建Echarts图表
              //环形嵌套图

              //饼图
              if (window.isMobile) {
                echartsPlugin.createMobilePieChart('mecharts2', dataList);
              } else {
                echartsPlugin.createPieChart('echarts2', dataList);
              }
            }
          });
        },
        //历史行情，切换时间
        changeDateHistory: function (value) {
          this.dateOptionHistory = value;

          if (value == '1M') {
            //this.historyFrequencyType = 'day';
            this.startAndEndTimeHistory = [ultls.getPreMonthDay(), ultls.getTodayValue()];
            this.queryIndexHistoryList();
          } else if (value == '6M') {
            //this.historyFrequencyType = 'day';
            this.startAndEndTimeHistory = [ultls.getSixMonthDay(), ultls.getTodayValue()];
            this.queryIndexHistoryList();
          } else if (value == '1Y') {
            //this.historyFrequencyType = 'day';
            this.startAndEndTimeHistory = [ultls.getPreYearDay(), ultls.getTodayValue()];
            this.queryIndexHistoryList();
          } else if (value == '2Y') {
            //this.historyFrequencyType = 'day';
            this.startAndEndTimeHistory = [ultls.getTwoYearDay(), ultls.getTodayValue()];
            this.queryIndexHistoryList();
          } else if (value == '3Y') {
            //this.historyFrequencyType = 'day';
            this.startAndEndTimeHistory = [ultls.getThreeYearDay(), ultls.getTodayValue()];
            this.queryIndexHistoryList();
          } else {
            //this.historyFrequencyType = 'day';
            this.startAndEndTimeHistory = [];
            this.queryIndexHistoryList();
          }
        },
        //历史行情，切换频次
        clickHistoryFrequency: function (val) {
          if (this.historyFrequencyType == val) return;

          this.historyFrequencyType = val;

          this.queryIndexHistoryList();
        },
        // 历史行情-下载
        historyDownload() {
          // 国证恒生指数走这个接口
          if (this.isHsiIndex) {
            this.hsiHistoryDownload();
            return;
          }
          // 其他指数保持原逻辑
          var options = {
            url: window.hqUrl + '/market/market/downloadDailyMarketExcel',
            method: 'post',
            params: {
              indexCode,
              startDate: this.startAndEndTimeHistory[0] || '',
              endDate: this.startAndEndTimeHistory[1] || '',
              frequency: this.historyFrequencyType
            }
          };
          ultls.download(options);
        },
        // 历史行情tab-查询列表数据
        queryIndexHistoryList: function () {
          var self = this;
          self.historyLoading = true;
          // 如果是恒生指数，走这个逻辑
          if (self.isHsiIndex) {
            this.queryHsiHistoryList(self);
            return;
          }
          // 其他指数走原逻辑
          $.ajax({
            url: window.hqUrl + '/market/market/getIndexDailyDataWithDataFormat',
            type: 'get',
            data: {
              indexCode: indexCode,
              startDate: self.startAndEndTimeHistory[0] || '',
              endDate: self.startAndEndTimeHistory[1] || '',
              frequency: self.historyFrequencyType
            },
            dataType: 'json'
          }).done(function (res) {
            if (res.code == 200) {
              self.historyLoading = false;
              self.historyAllData = [];
              self.historyPage = 1;

              var dataList = [],
                arrayList = res.data.data;
              i = 0;

              if (arrayList.length > 0) {
                for (i; i < arrayList.length; i++) {
                  var obj = {
                    date: arrayList[i][0],
                    open: arrayList[i][3],
                    close: arrayList[i][5],
                    high: arrayList[i][2],
                    low: arrayList[i][4],
                    percent: arrayList[i][7],
                    amount: arrayList[i][8],
                    volume: arrayList[i][9]
                  };
                  dataList.push(obj);
                }

                self.historyAllData = dataList;

                if (dataList.length < 20) {
                  self.historyShowData = dataList;
                } else {
                  self.historyShowData = dataList.slice(0, 20);
                }

                self.historyAllDataTotal = res.total;
              } else {
                self.historyAllData = [];
                self.historyShowData = [];
                self.historyAllDataTotal = 0;
              }
            }
          });
        },
        //获取相关产品数据
        queryFundList: function (pageNo, pageSize) {
          var self = this;
          self.fundLoading = true;

          if (pageNo == 1) {
            self.fundList = [];
            self.fundTotal = 0;
          }
          $.ajax({
            url: '/info/fund',
            type: 'get',
            data: {
              indexCode: indexCode,
              pageNum: pageNo,
              rows: pageSize
            },
            dataType: 'json'
          }).done(function (res) {
            if (res.code == 200 && !$.isEmptyObject(res.data)) {
              self.fundLoading = false;
              //self.fundList = [];
              var dataList = [],
                arrayList = res.data.rows;
              i = 0;

              if (res.data.msg) {
                self.fundUpdateTime = res.data.msg;
              }

              for (i; i < arrayList.length; i++) {
                var obj = {
                  fundCode: ultls.fundNameFormate(arrayList[i]['fundCode']),
                  fundName: arrayList[i]['fundName'],
                  fundDate: ultls.formateDate(arrayList[i]['fundDate']),
                  fundScale: arrayList[i]['fundScale'],
                  fundType: arrayList[i]['fundType'],
                  foundLocation: arrayList[i]['foundLocation'],
                  enFoundLocation: arrayList[i]['enFoundLocation'],
                  fundManger: arrayList[i]['fundManger'],
                  fundIndexCode: arrayList[i]['fundIndexCode'],
                  fundNetValue: arrayList[i]['fundNetValue'],
                  enFundManger: arrayList[i]['enFundManger'],
                  enFundName: ultls.fundNameFormate(arrayList[i]['enFundName']),
                  enFundType: arrayList[i]['enFundType']
                };
                dataList.push(obj);
              }

              if (pageNo == 1) {
                self.fundList = dataList;
              } else {
                self.fundList = self.fundList.concat(dataList);
              }
              self.fundTotal = res.data.total;
            } else {
              self.fundLoading = false;
              self.tabsList.pop();
            }
          });
        },
        //样本详情月份切换
        changeSampleDate: function () {
          this.querySampleList(1, 20);
        },
        // 获取样本详情列表数据
        querySampleList: function (pageNo, pageSize, isFirstCall) {
          var self = this;
          self.sampleLoading = true;
          if (pageNo == 1) {
            self.samplePage = 1;
            self.sampleList = [];
            self.sampleTotal = 0;
          }
          // 如果是恒生指数，走这个逻辑
          if (self.isHsiIndex) {
            this.queryHsiSampleList(pageNo, pageSize, self);
            return;
          }
          // 其他指数走原逻辑
          // 1、普通指数 2、债券指数 3、基金指数
          // 老版债券指数/bondIndexSample/query、新版债券指数/sample-detail/bondsDetail
          var urlList = ['/sample-detail/detail', '/sample-detail/bondsDetail', '/fundIndexSample/detail'],
            currentUrl = '';

          if (self.newIndexType) {
            currentUrl = urlList[1];
          } else if (self.indexType == 107 || self.indexType == 207) {
            currentUrl = urlList[2];
          } else {
            currentUrl = urlList[0];
          }
          var data = {
            indexcode: indexCode,
            // dateStr: self.sampleDate,
            dateStr: self.currDate,
            pageNum: pageNo,
            rows: pageSize
            // t:new Date().getTime()
          };
          if (isFirstCall && isFirstCall == '1') {
            data = {
              indexcode: indexCode,
              // dateStr: self.sampleDate,
              dateStr: self.currDate,
              pageNum: pageNo,
              rows: pageSize,
              isFirstCall: isFirstCall
              // t:new Date().getTime()
            };
          }
          $.ajax({
            url: currentUrl,
            type: 'get',
            data: data,
            dataType: 'json'
          }).done(function (res) {
            self.sampleLoading = false;
            if (isFirstCall == '1' && res.code == 200 && res.sampleDate && !!res.sampleDate) {
              self.currDate = res.sampleDate;
              self.querySampleList(1, 20);
              return;
            }
            if (res.code == 200 && res.data && res.data.rows) {
              var arr = self.currDate.split('-');
              if (res.data.total == 0 && (arr[1] == '06' || arr[1] == '12')) {
                var tempArr = self.currDate.split('-');
                var tempData = '';
                if (arr[1] == '06') {
                  tempData = tempArr[0] + '-05';
                } else {
                  tempData = tempArr[0] + '-11';
                }
                self.currDate = tempData;
                self.querySampleList(1, 20);
              } else {
                //接口数据错误，做条件判断
                // self.sampleTotal = (self.indexType == 106 || self.indexType == 206) ? res.data.total : res.total;
                self.sampleTotal = res.total;
                if (pageNo == 1) {
                  self.sampleList = res.data.rows;
                } else {
                  self.sampleList = self.sampleList.concat(res.data.rows);
                }
              }
            } else if (res.code == 200 && res.data == null) {
              self.$refs.myDate.userInput = null;
              if (self.currDate === self.sampleDate) return;
              self.currDate = self.sampleDate;
              self.querySampleList(1, 20);
            }
          });
        },
        // 获取调整分析指标
        getIndexAnalysis(code) {
          const self = this;
          $.ajax({
            url: '/indexAnalysis/getIndexAnalysis',
            type: 'get',
            data: {
              indexCode: code
            },
            dataType: 'json'
          }).done(function (res) {
            const { code, data } = res;
            if (code == 200 && data) {
              // 有数据展示
              self.indexAnalysisFlag = true;
              const indexAnalysisItem = {
                totalMarketValue: '-',
                yieldMaturity: '-',
                modifiedDuration: '-',
                convexity: '-',
                avgResidualMaturity: '-',
                genDate: '-'
              };
              for (let key in indexAnalysisItem) {
                if (data[key] || data[key] === 0) {
                  if (key === 'totalMarketValue') {
                    const nowIndexAnalysisItem = data[key] / 100000000;
                    indexAnalysisItem[key] = nowIndexAnalysisItem.toFixed(2);
                  } else {
                    indexAnalysisItem[key] = key === 'yieldMaturity' ? data[key] + '%' : data[key];
                  }
                }
              }
              self.indexAnalysis.push(indexAnalysisItem);
            }
          });
        },
        // 请求指数简介、指数编制
        queryIndexInfo: function (code) {
          var self = this;
          $.ajax({
            url: '/index-intro',
            type: 'get',
            data: { indexcode: code },
            dataType: 'json'
          }).done(function (res) {
            if (res.code == 200) {
              if (res.data) {
                self.indexInfo = res.data;
              } else {
                self.indexInfo = {
                  jsjj: '',
                  xyfw: '',
                  jsfs: '',
                  qzsx: '',
                  typl: ''
                };
              }
            }
          });
        },
        /*页码操作函数*/
        getPageCountRequest: function () {
          var scrollT = $(document).scrollTop(),
            clientH = $(window).height(),
            eleOffsetT = $('.scroll-load').offset().top;

          if (this.activeName == 1) {
            eleOffsetT = $('.scroll-load-1').offset().top;
          } else if (this.activeName == 2 || this.mobileActiveName == 3) {
            const showLoad2 = $('.scroll-load-2').filter((index, dom) => {
              return dom.offsetHeight > 0;
            });
            eleOffsetT = showLoad2.offset().top;
          } else {
            eleOffsetT = $('.scroll-load-3').offset().top;
          }

          if (scrollT > 400) {
            this.isFix = true;
          } else {
            this.isFix = false;
          }

          if (window.isMobile) {
            if (scrollT > eleOffsetT - clientH) {
              //历史行情分页
              if (this.mobileActiveName == 2) {
                if (this.historyShowData.length < this.historyAllData.length) {
                  this.historyShowData = this.historyAllData.slice(0, this.historyPage * 20);
                  this.historyPage += 1;
                }
              }
              //样本详情分页
              if (this.mobileActiveName == 3) {
                if (this.sampleLoading) {
                  return;
                }
                if (this.samplePage < this.sampleTotal / 20) {
                  this.samplePage += 1;
                  this.querySampleList(this.samplePage, 20);
                }
              }
              //相关产品分页
              if (this.mobileActiveName == 4) {
                if (this.fundLoading == 'true') {
                  return;
                }

                if (this.fundList.length < this.fundTotal) {
                  this.fundPage += 1;
                  this.queryFundList(this.fundPage, 20);
                }
              }
            }
          } else {
            if (scrollT > eleOffsetT - clientH) {
              //历史行情分页
              if (this.activeName == 1) {
                if (this.historyShowData.length < this.historyAllData.length) {
                  this.historyShowData = this.historyAllData.slice(0, this.historyPage * 20);
                  this.historyPage += 1;
                }
              }
              //样本详情分页
              if (this.activeName == 2) {
                if (this.sampleLoading) {
                  return;
                }
                if (this.samplePage < this.sampleTotal / 20) {
                  this.samplePage += 1;
                  this.querySampleList(this.samplePage, 20);
                }
              }
              //相关产品分页
              if (this.activeName == 3) {
                if (this.fundLoading == 'true') {
                  return;
                }

                if (this.fundList.length < this.fundTotal) {
                  this.fundPage += 1;
                  this.queryFundList(this.fundPage, 20);
                }
              }
            }
          }
        },
        //获取阶段性收益
        queryIncomeInfo: function (code) {
          var self = this;
          $.ajax({
            url: '/index-income',
            type: 'get',
            data: {
              indexcode: code
            },
            dataType: 'json'
          }).done(function (res) {
            if (res.code == 200 && res.data) {
              self.incomeDate = res.data.deadline;
              self.incomeList.push(res.data);
            }
          });
        },
        // 查询相关新闻
        queryNewsList: function (code) {
          var self = this;
          $.ajax({
            url: '/info/news',
            type: 'get',
            data: {
              newsIndexCode: code
            },
            dataType: 'json'
          }).done(function (res) {
            if (res.code == 200) {
              self.newsList = res.data;
            }
          });
        },
        // 给移动端页面顶部写入指数名称和交易时间
        setDomText({ domId, text }) {
          // 非移动端不用处理
          if (!window.isMobile) {
            return;
          }
          // 由于顶部dom渲染时机不确定，所以在3秒内遍历加载
          let count = 0;
          const funWrite = () => {
            const dom = document.getElementById(domId);
            if (dom) {
              // 移动端不展示--，转成空字符串
              dom.innerText = text === '--' ? '' : text;
              return;
            }
            if (count < 10) {
              count++;
              setTimeout(() => {
                funWrite();
              }, 300);
            }
          };
          funWrite();
        },
        // 查询所有线上指数
        queryAllOnlineIndex() {
          $.ajax({
            url: '/index/selectOnlineIndexCode',
            type: 'get',
            async: true,
            dataType: 'json'
          }).done(res => {
            if (res.code === 200 && Array.isArray(res.data) && res.data.length > 0) {
              this.allOnlineIndexList = res.data.map(item => ({ indexCode: item.indexcode, indexName: item.indexname, indexEName: item.indexename }));
            }
          });
        },
        // 指数对比-键盘精灵查询指数列表
        showKeyboardInfo: function (queryStr, callback) {
          if (!queryStr) {
            callback([]);
            return;
          }
          var self_this = this;
          $.ajax({
            type: 'POST',
            url: '/keyBoard/queryKeyboardInfo',
            datType: 'JSON',
            data: { keyWord: self_this.searchVal },
            success: function (res) {
              if (res.code == 200) {
                callback(res.data);
              } else {
                self_this.$message({
                  message: res.message,
                  type: 'info',
                  duration: 5000
                });
              }
            },
            error: function (err) {
              self_this.$message({
                message: err.message,
                type: 'info',
                duration: 5000
              });
            }
          });
        },
        // 指数对比-键盘精灵选择
        selectOtherIndex: function (item) {
          var self = this;
          var flag = false;
          if (item.indexCode == indexCode) {
            this.$message({ showClose: true, type: 'warning', message: window.isEN ? 'Index Existed' : '指数已存在' });
            return;
          }
          $.each(this.contrastIndexList, function (index, obj) {
            if (item.indexCode == obj.indexCode) {
              self.$message({
                showClose: true,
                type: 'warning',
                message: window.isEN ? 'Index Existed' : '指数已存在'
              });
              flag = true;
            }
          });
          if (this.contrastIndexList.length > 3) {
            this.$message({
              showClose: true,
              type: 'warning',
              message: window.isEN ? 'Exceed the upper limit' : '超过对比指数上限'
            });
            return;
          }

          if (flag) {
            return;
          }
          this.contrastIndexList.push(item);
          this.queryIncomeInfo(item.indexCode);
        },
        //删除对比指数
        deleteContrastItem: function (index) {
          this.contrastIndexList.splice(index, 1);
          this.incomeList.splice(index + 1, 1);
        },
        // 样本详情tab-点击下载按钮
        downloadSample: function () {
          // 国证恒生指数走这个逻辑
          var self = this;
          if (self.isHsiIndex) {
            self.hsiSampleDownload(self);
            return;
          }
          // 其他指数走这个逻辑：1、普通指数 2、债券指数 3、基金指数
          var urlList = ['/sample-detail/download', '/bondIndexSample/download', '/fundIndexSample/download'],
            currentUrl = '';

          if (self.newIndexType) {
            currentUrl = urlList[1];
          } else if (self.indexType == 107 || self.indexType == 207) {
            currentUrl = urlList[2];
          } else {
            currentUrl = urlList[0];
          }

          var options = {
            url: currentUrl,
            method: 'get',
            params: {
              indexcode: indexCode,
              dateStr: self.newIndexType ? this.currDate : this.sampleDate
            }
          };
          ultls.download(options);
        },
        //历史调样下载
        downloadAdjustment: function () {
          var self = this;
          //1、普通指数 2、债券指数 3、基金指数
          var urlList = ['/sample-detail/download-adjustment', '/bondIndexSample/download-adjustment', '/fundIndexSample/download-adjustment'],
            currentUrl = '';

          if (self.newIndexType) {
            currentUrl = urlList[1];
          } else if (self.indexType == 107 || self.indexType == 207) {
            currentUrl = urlList[2];
          } else {
            currentUrl = urlList[0];
          }
          var options = {
            url: currentUrl,
            method: 'get',
            params: {
              indexcode: indexCode,
              dateStr: this.currDate
            }
          };
          if (!self.newIndexType && !(self.indexType == 107 || self.indexType == 207)) {
            options = {
              url: currentUrl,
              method: 'get',
              params: {
                indexcode: indexCode
              }
            };
          }
          ultls.download(options);
        },
        //历史样本下载
        downloadHistory: function () {
          var self = this;
          //1、普通指数 2、债券指数 3、基金指数
          var urlList = ['/sample-detail/download-history', '/bondIndexSample/download-history', '/fundIndexSample/download-history'],
            currentUrl = '';

          if (self.newIndexType) {
            currentUrl = urlList[1];
          } else if (self.indexType == 107 || self.indexType == 207) {
            currentUrl = urlList[2];
          } else {
            currentUrl = urlList[0];
          }
          var options = {
            url: currentUrl,
            method: 'get',
            params: {
              indexcode: indexCode,
              dateStr: this.currDate
            }
          };
          if (!self.newIndexType && !(self.indexType == 107 || self.indexType == 207)) {
            options = {
              url: currentUrl,
              method: 'get',
              params: {
                indexcode: indexCode
              }
            };
          }
          ultls.download(options);
        },
        clickFullScreen: function () {
          this.fullScreenDialog = true;
        },
        //技术指标修改
        indicatorsChange: function (value) {
          // if (value == this.indicatorsValue) return;
          if (value == 'MACD') {
            timeChart.series[2].update(macdOption);
            timeChart2.series[2].update(macdOption);
          } else if (value == 'BOLL') {
            timeChart.series[2].update(bollOption);
            timeChart2.series[2].update(bollOption);
          } else if (value == 'ROC') {
            timeChart.series[2].update(rocOption);
            timeChart2.series[2].update(rocOption);
          } else if (value == 'RSI') {
            timeChart.series[2].update(rsiOption);
            timeChart2.series[2].update(rsiOption);
          }
        },
        blurHotIndexBox: function () {
          var self = this;
          setTimeout(function () {
            self.showHotIndexBox = false;
          }, 250);
        },
        // 移动端切换导航
        mobileMainTabClick: function (value) {
          // 切换到指数表现
          if (value.name == 1 && !this.isHsiIndex) {
            this.queryMarketDistribute(indexCode);
          }
          // 切换到历史行情
          if (value.name == 2) {
            if (this.historyAllData.length > 0) return;
            this.startAndEndTimeHistory = [ultls.getPreMonthDay(), ultls.getTodayValue()];
            this.queryIndexHistoryList();
          }
          // 切换到样本详情
          if (value.name == 3) {
            if (this.sampleList.length > 0) return;
            this.querySampleList(1, 20);
          }
          // 切换到相关产品
          if (value.name == 4) {
            if (this.fundList.length > 0) return;
            this.queryFundList(1, 20);
          }
        },
        // 延迟一定毫秒的异步函数
        promiseSomeTime(time) {
          return new Promise(resolve => {
            setTimeout(() => {
              resolve(true);
            }, time);
          });
        }
      }
    });

    // 请求指数对比数据
    var queryIndexContrastData = function (strCodes) {
      // 将strCodes拆成数组
      const arrCode = strCodes.split(';').filter(item => !!item);
      let arrContrastData = []; // 根据strCodes的指数顺序，准备一个数组来收集完整的渲染数据
      let requestDoneNum = 0; // 请求完成的次数，当与arrContrastData.length长度一样时则表示请求全部完成
      let arrOnlineCode = []; // 收集线上指数code
      let arrOfflineCode = []; // 收集线下指数code
      arrCode.forEach(code => {
        const findOnline = vm.allOnlineIndexList.find(oItem => oItem.indexCode === code);
        const objItem = { indexCode: code, indexName: '', indexEName: '', dataList: [] };
        // 线上指数的指数名和英文名从这里取，线下的从下面的接口取
        if (findOnline) {
          arrOnlineCode.push(code);
          Object.assign(objItem, { indexName: findOnline.indexName || code, indexEName: findOnline.indexEName || code });
        } else {
          arrOfflineCode.push(code);
        }
        // indexCode用于标记指数；dataList用于存储图表需要的数据
        arrContrastData.push(objItem);
      });

      // 先处理一下时间范围
      const dateRange = vm.startAndEndTime && vm.startAndEndTime.length > 0 ? vm.startAndEndTime : [vm.jrStartDate, vm.jrEndDate];
      // 按线上线下区分开来，线上走NSDK
      if (arrOnlineCode.length > 0) {
        const params = {
          indexCode: '',
          arrField: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'PreClose'],
          dateRange,
          period: 'day'
        };
        arrOnlineCode.forEach(async codeItem => {
          params.indexCode = codeItem;
          const { code, data, response } = await NsdkRequest.SDK_REQUEST_KLINE(params).catch(err => err);
          const findContrast = arrContrastData.find(cItem => cItem.indexCode === codeItem);
          // 请求失败则存储空数据
          if (code !== 200) {
            findContrast.dataList = [];
          } else {
            const arrDate = data[0];
            const arrOpen = data[1];
            const arrHigh = data[2];
            const arrLow = data[3];
            const arrClose = data[4];
            const arrVolume = data[5];
            const arrAmount = data[6];
            const arrPreClose = data[7];
            // 这个值原本取arrPreClose[0]，但是它有可能是0，比如第一天上市的股，前一天的收盘价就是0，所以这里处理为找一下第一个不为0的收盘价
            const firstPreClose = arrPreClose.find(pItem => !!pItem);
            // 请求成功则组装成跟接口一样的数据格式，挑日期数组来做遍历（挑哪个都行，他们长度都一样）
            findContrast.dataList = arrDate.map((date, index) => {
              // NSDK给的日期格式是‘YYYYMMDD’，要转成'YYYY-MM-DD hh:mm:ss'，保持跟后端接口返回的一致
              const strDate = date.toString();
              const ymdhms = `${strDate.substring(0, 4)}-${strDate.substring(4, 6)}-${strDate.substring(6, 8)} 00:00:00`;
              const dateTime = Date.parse(ymdhms);
              // 第11个值NSDK没有，这里根据后端提供的公式计算：第一个数据固定为0，后面的为：（当日收盘/第一个昨日收盘价）- 1 后再取十位小数
              const perData = index === 0 ? 0 : Number((arrClose[index] / firstPreClose - 1).toFixed(10));
              const arrResult = [
                dateTime, // 0时间戳
                arrClose[index], // 1当前点位
                arrHigh[index], // 2最高价
                arrOpen[index], // 3开盘价
                arrLow[index], // 4最低价
                arrClose[index], // 5收盘价
                arrClose[index] - arrPreClose[index], // 6涨跌（当日收盘价-昨日收盘价）
                (arrClose[index] - arrPreClose[index]) / arrClose[index], // 7涨跌幅百分比
                arrAmount[index], // 8成交额
                arrVolume[index], // 9成交量
                (arrClose[index] + arrPreClose[index]) / 2, // 10均价（NSDK没这个值，暂时取（当日收盘价 + 昨日收盘价）/2）
                perData // 11收盘差值比
              ];
              return arrResult;
            });
          }
          requestDoneNum += 1;
          if (requestDoneNum === arrContrastData.length) {
            createContrastChart(arrContrastData);
          }
        });
      }
      // 线下走数据库接口
      if (arrOfflineCode.length > 0) {
        $.ajax({
          url: window.hqUrl + '/market/market/getIndexContrastDataByIndexCodes',
          type: 'get',
          dataType: 'json',
          async: true,
          data: { indexCodes: arrOfflineCode.join(';'), startDate: dateRange[0], endDate: dateRange[1] }
        }).done(res => {
          // 成功返回数组数据时
          if (res.code === 200 && Array.isArray(res.data) && res.data.length > 0) {
            // 为了确保arrContrastData里一定要收集到数据（不论接口是否有返回正确的指数数据），这里要用arrOfflineCode来做遍历头
            arrOfflineCode.forEach(code => {
              // 在返回的数据中找对应指数
              const findData = res.data.find(rItem => rItem.indexCode === code);
              const findContrast = arrContrastData.find(cItem => cItem.indexCode === code);
              // 能找到，就更新指数名称、赋值渲染数据
              if (findData) {
                Object.assign(findContrast, { indexName: findData.indexName, indexEName: findData.indexEName, dataList: findData.data });
              } else {
                // 没找到就把code当名称、赋值空数据
                Object.assign(findContrast, { indexName: code, indexEName: code, dataList: [] });
              }
            });
          } else {
            // 未返回数据时，填充空数据进去
            arrOfflineCode.forEach(code => {
              const findContrast = arrContrastData.find(cItem => cItem.indexCode === code);
              findContrast.dataList = [];
            });
          }
          requestDoneNum += arrOfflineCode.length;
          if (requestDoneNum === arrContrastData.length) {
            createContrastChart(arrContrastData);
          }
        });
      }
    };
  }
);
