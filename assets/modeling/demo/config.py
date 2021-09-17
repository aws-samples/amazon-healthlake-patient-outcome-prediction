dashboard_style = '''
<style type="text/css">
    .site-header {
        background-color: rgba(0, 0, 0, .85); 
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        backdrop-filter: saturate(180%) blur(20px);
    }
    .site-header a {
        color: #999;
        transition: ease-in-out color .15s;
    }
    .site-header a:hover {
        color: #fff;
        text-decoration: none;
    }
    .border-top { border-top: 1px solid #e5e5e5; }
    .border-bottom { border-bottom: 1px solid #e5e5e5; }

    .box-shadow { box-shadow: 0 .25rem .75rem rgba(0, 0, 0, .05); }
</style>
'''

sunburst_html = '''
<div class="position-relative overflow-hidden p-3 p-md-4 m-md-3 text-center bg-light">
    <div class="my-3 p-3 overflow-hidden">
      <h2 class="display-5">Current Therapeutical hierarchy and Population Size</h2>
      <div id="sunburst" class="bg-light box-shadow mx-auto overflow-hidden">
        <script>
        
        var sunburst_data = {
            "name": "ICD",
            "children": [{
               "name": "ChronicDisease",
               "children": [
                {
                 "name": "J45-Asthma",
                 "children": [
                  {"name": "J45.2x Mild intermittent", "size": 50},
                  {"name": "J54.3x Mild persistent", "size": 50},
                  {"name": "J45.4x Moderate persistent", "size": 100},
                  {"name": "J45.5x Severe persistent", "size": 200},
                  {"name": "J45.9x Other and unspecified", "size": 300}
                 ]
                },
                {
                 "name": "J44-COPD",
                 "children": [ 
                  {"name": "J44.0 COPD with acute infection", "size": 100},
                  {"name": "J44.1 COPD with acute exacerbation", "size": 100},
                  {"name": "J44.9 COPD, unspecified", "size": 50}
                 ]
                },
                {
                 "name": "E08-Diabetes",
                 "children": [
                  {"name": "E08.0x Diabetes hyperosmolarity", "size": 100},
                  {"name": "E08.1x Diabetes ketoacidosis", "size": 800},
                  {"name": "E08.2x Diabetes kidney complications", "size": 300},
                  {"name": "E08.3x Diabetes ophthalmic complications", "size": 200},
                  {"name": "E08.4x Diabetes neurological complications", "size": 400},
                  {"name": "E08.5x Diabetes circulatory complications", "size": 300},
                  {"name": "E08.6x Diabetes other specified complications", "size": 100},
                  {"name": "E08.8 Diabetes unspecified complications", "size": 800}
                 ]
                },
                {
                 "name": "hypertension",
                 "children": [
                  {"name": "I10.x Essential hypertension", "size": 1000},
                  {"name": "I15.x Secondary hypertension", "size": 800},
                  {"name": "I27.0 Primary pulmonary hypertension", "size": 600},
                  {"name": "I27.2 Other secondary pulmonary hypertension", "size": 600},
                  {"name": "O10 O16", "size": 100}
                 ]
                }
               ]},
               {
               "name": "HF",
               "children": [
                  {"name": "I11.0 heart disease with heart failure", "size": 100},
                  {"name": "I13.0 heart and CKD", "size": 600},
                  {"name": "I13.1x heart and CKD without heart failure", "size": 300},
                  {"name": "I13.2 heart and CKD5 or ESRD", "size": 700},
                  {"name": "I50.x Heart failure", "size": 900}
                  ]

               }
              ]
            };
        var width = 800,
            height = 800,
            radius = (Math.min(width, height) / 2) - 150;

        var formatNumber = d3.format(",d");

        var x = d3.scale.linear()
            .range([0, 2 * Math.PI]);

        var y = d3.scale.sqrt()
            .range([0, radius]);

        var color = d3.scale.category20c();

        var partition = d3.layout.partition()
            .value(function(d) { return d.size; });

        var arc = d3.svg.arc()
            .startAngle(function(d) { return Math.max(0, Math.min(2 * Math.PI, x(d.x))); })
            .endAngle(function(d) { return Math.max(0, Math.min(2 * Math.PI, x(d.x + d.dx))); })
            .innerRadius(function(d) { return Math.max(0, y(d.y)); })
            .outerRadius(function(d) { return Math.max(0, y(d.y + d.dy)); });

        var svg = d3.select("#sunburst").append("svg")
            .attr("width", width)
            .attr("height", height)
          .append("g")
            .attr("transform", "translate(" + width / 2 + "," + (height / 2) + ")");

        function generate_sunburst(root) {
          var g = svg.selectAll("g")
              .data(partition.nodes(root))
            .enter().append("g");

          var path = g.append("path")
            .attr("d", arc)
            .attr("stroke", "#fff")
            .style("fill", function(d) { return color((d.children ? d : d.parent).name); })
            .on("click", click);

          var text = g.append("text")
            .attr("transform", function(d) { return "rotate(" + computeTextRotation(d) + ")"; })
            .attr("x", function(d) { return y(d.y); })
            .attr("dx", "6") // margin
            .attr("dy", ".35em") // vertical-align
            .text(function(d) { return d.name; });

          function click(d) {
            text.transition().attr("opacity", 0);

            path.transition()
              .duration(750)
              .attrTween("d", arcTween(d))
              .each("end", function(e, i) {
                  // check if the animated element's data e lies within the visible angle span given in d
                  if (e.x >= d.x && e.x < (d.x + d.dx)) {
                    // get a selection of the associated text element
                    var arcText = d3.select(this.parentNode).select("text");
                    // fade in the text element and recalculate positions
                    arcText.transition().duration(750)
                      .attr("opacity", 1)
                      .attr("transform", function() { return "rotate(" + computeTextRotation(e) + ")" })
                      .attr("x", function(d) { return y(d.y); });
                  }
              });
          }
        }
        generate_sunburst(sunburst_data);

        d3.select(self.frameElement).style("height", height + "px");

        function arcTween(d) {
          var xd = d3.interpolate(x.domain(), [d.x, d.x + d.dx]),
              yd = d3.interpolate(y.domain(), [d.y, 1]),
              yr = d3.interpolate(y.range(), [d.y ? 20 : 0, radius]);
          return function(d, i) {
            return i
                ? function(t) { return arc(d); }
                : function(t) { x.domain(xd(t)); y.domain(yd(t)).range(yr(t)); return arc(d); };
          };
        }

        function computeTextRotation(d) {
          return (x(d.x + d.dx / 2) - Math.PI / 2) / Math.PI * 180;
        }
        </script>
      </div>
    </div>
</div>   
'''