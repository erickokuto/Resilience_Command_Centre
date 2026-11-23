library(shiny)
library(DT)
library(ggplot2)
options(shiny.maxRequestSize=30*1024^2)
root <- normalizePath(getwd(),winslash='/')
python <- Sys.getenv('SUPPLY_PYTHON',unset=if(.Platform$OS.type=='windows') 'python' else 'python3')
readcsv <- function(p) read.csv(p,check.names=FALSE,colClasses=c(po_id='character',item_id='character'),stringsAsFactors=FALSE,na.strings=c('','NA'),fileEncoding='UTF-8-BOM')
fmt <- function(x,d=1) if(length(x)==0||!is.finite(x)) '—' else formatC(x,format='f',digits=d,big.mark=',')
pct <- function(x) paste0(fmt(100*x),'%')
euro <- function(x) paste0('EUR ',fmt(x,0))
key <- function(x) paste(x$po_id,x$item_id,sep=' / ')
kpi <- function(a,b) div(class='kpi',div(class='labelx',a),div(class='valuex',b))
panel <- function(...) div(class='panelx',...)
ui <- fluidPage(tags$head(tags$style(HTML('body{background:#eef2f5;color:#19344a;font-family:Arial,sans-serif}.banner{background:#15394d;color:white;padding:20px 24px;margin:0 -15px 18px}.banner h2{margin:0 0 8px}.panelx{background:white;border:1px solid #dce4e9;border-radius:7px;padding:16px;margin:12px 0}.kpis{display:flex;flex-wrap:wrap;gap:12px}.kpi{background:white;border:1px solid #dce4e9;border-radius:7px;flex:1 1 160px;padding:14px}.labelx{font-size:12px;color:#526776}.valuex{font-size:25px;font-weight:bold;margin-top:6px}.note{color:#506878;font-size:13px}.nav-tabs{margin-bottom:12px}.btn-primary{background:#176e76;border-color:#176e76}'))),
 div(class='banner',h2('Supply Chain Resilience | Command Centre'),div('German manufacturing · Operational decision support · Synthetic proof of concept')),
 sidebarLayout(sidebarPanel(width=3,
  h4('Active order batch'),fileInput('upload','Browse: Scoring.csv or another order CSV',accept='.csv'),downloadButton('upload_example','Download sample order CSV'),p('Try Upload_Example.csv (12 orders) for a quick demonstration. Use data/Scoring.csv (1,000 open orders) or a compatible batch of at most 1,000 rows. Do not upload summary, supplier master, external signals or prediction exports.'),actionButton('run','Validate and analyse upload',class='btn-primary'),actionButton('demo','Restore demonstration batch'),verbatimTextOutput('status'),hr(),
  selectInput('plant','Plant','All'),selectInput('transport','Transport',c('All','Air','Rail','Road','Sea')),selectInput('critical','Material criticality',c('All','High','Medium','Low')),actionButton('reset_filters','Reset filters and thresholds'),sliderInput('amber','Review threshold',0,0.95,.4,.05),sliderInput('red','High-risk threshold',.05,1,.75,.05),selectInput('level','Nominal interval',c('80','90'),selected='90'),selectInput('prob','Delay probability',c('Original'='raw_late_probability','Calibrated'='calibrated_late_probability')),downloadButton('export','Export filtered orders'),p(class='note','Batch limit: 1,000 orders. Data are evaluated at order creation; this is not a live shipment tracker.')),
 mainPanel(width=9,tabsetPanel(
 tabPanel('Executive overview',h3('Operational outlook'),p('Use the left-hand controls to filter all operational views. Costs and exposure are simulated; not realised savings.'),uiOutput('kpis'),panel(h4('Delay risk distribution'),plotOutput('riskplot',height=270)),panel(h4('Recommended interventions'),plotOutput('actions',height=300)),panel(h4('Risk by plant and transport'),p('Use the sidebar filters to narrow the active batch.'),fluidRow(column(6,plotOutput('plant_risk')),column(6,plotOutput('mode_risk')))),panel(h4('Highest-priority orders'),DTOutput('priority'))),
 tabPanel('Early warnings',panel(h3('Risk alert queue'),p('High risk and review thresholds are adjustable in the sidebar. Anomaly flags are separate from delay probability. Uncheck the box to see all matching orders.'),checkboxInput('alerts','Show review/high risk or anomalous orders only',TRUE),DTOutput('alerts_table')),panel(h4('Forecast uncertainty'),plotOutput('intervals',height=340))),
 tabPanel('Supplier intelligence',panel(h3('Supplier exposure in active batch'),p('Clusters describe logistics exposure, not supplier quality. Unseen suppliers may have no cluster.'),DTOutput('suppliers')),panel(h4('Supplier risk and anomaly patterns'),plotOutput('supplier_risk'),plotOutput('anomaly_hist')),panel(h4('Unusual orders'),DTOutput('anomalies'))),
 tabPanel('Inventory & sourcing',uiOutput('inventory_kpis'),panel(h3('Order-level procurement recommendations'),p('Compare current inventory coverage, proposed buffer, alternative capacity, expected cost and remaining shortage exposure. Select an order below to inspect the recommendation.'),DTOutput('decisions')),panel(h4('Cost and inventory exposure'),fluidRow(column(6,plotOutput('savings_plot')),column(6,plotOutput('inventory_plot',click='inventory_click')))),panel(h4('Inspect an order'),p('Select a purchase order line or click an inventory-coverage scatterplot point above.'),selectInput('order','Purchase order line',choices=NULL),uiOutput('detail'),p(class='note','Alternative sources are simulated quotes. Confirm capacity, approvals, qualification and cost before action. Inventory buffers are independent per order.'))),
 tabPanel('Scenario simulator',panel(h3('Stress test of fixed base-case policies'),radioButtons('stress','Scenario',c('Base'='0','Moderate (+7 sea / +3 other days; demand +15%)'='1','Severe (+14 sea / +6 other days; demand +30%)'='2')),p('Alternative supplier delay is +3 or +6 days in moderate/severe conditions. Policies are not reoptimised under stress.'),uiOutput('scenario_kpis'),plotOutput('costplot',height=280),plotOutput('stockplot',height=280),plotOutput('stress_trend',height=300),DTOutput('scenario_table'),downloadButton('scenario_export','Export scenario comparison'))),
 tabPanel('Planner review',panel(h3('Human review and decision log'),p('Select an order, select a review decision, write a reason, and click Record review. Export the log before changing data or closing the application.'),p('A session-only planning record. No order, purchase or SAP update is performed.'),selectInput('review_order','Order line',choices=NULL),radioButtons('review_status','Decision',c('Investigate','Approve for planning','Reject recommendation')),textAreaInput('review_note','Planner notes',rows=3),actionButton('save_review','Record review'),DTOutput('review_log'),downloadButton('review_export','Export review log'))),

 tabPanel('Adaptive model refresh',panel(h3('Update predictive decision support from ONE uploaded dataset'),
  p('Completed orders are partitioned for training, validation and testing; open orders are forecast. Choose the percentages from the full labelled portion of this one file.'),
  fileInput('adaptive_file','Upload complete order-level CSV',accept='.csv'),
  downloadButton('adaptive_example','Download consolidated example'),
  fluidRow(column(4,sliderInput('adaptive_train','Training (%)',10,80,70,step=5)),
           column(4,sliderInput('adaptive_validation','Validation (%)',10,80,15,step=5)),
           column(4,uiOutput('adaptive_test_pct'))),
  selectInput('adaptive_method','Split method',c('Chronological (recommended)'='chronological','Random'='random')),
  actionButton('adaptive_go','Train and refresh risk priorities',class='btn-primary'),
  verbatimTextOutput('adaptive_status')),
  panel(h3('Independent evaluation of candidate refreshed models'),DTOutput('adaptive_metrics'),uiOutput('adaptive_summary')),
  panel(h3('Updated risk queue and suggested planning review'),p('These are advisory priorities from the newly trained predictive models. Cost savings, optimised buffers and scenario simulations in the other tabs still use the historical frozen models and must not be interpreted as updated by this step.'),
  DTOutput('adaptive_predictions'),downloadButton('adaptive_download','Export refreshed order priorities'))),
 tabPanel('Governance',panel(h3('Data and model governance'),p('Start with Restore demonstration batch. Use Browse only for order-level inputs. Use the filters and thresholds to prioritise risk; scenario results are simulated. Export review records before closing or changing batch.'),p('The original operational dashboard uses frozen analytical models and simulated scenarios. The separate Adaptive model refresh tab fits session-only predictive candidates; it does not replace historical procurement optimisation.'),p('Do not upload confidential enterprise data to public services. SAP BTP/IBP integration is not implemented. This application provides decision support, not automated procurement.'),p('Cost estimates exclude common base purchasing costs. Shared inventory, working capital, multi-period and supplier-wide constraints are not modelled.'),downloadButton('schema','Download data dictionary'),downloadButton('quality_export','Download validation report')))
 ))))
server <- function(input,output,session){

 # -------------------------------------------------------------------------
 # Adaptive model refresh: single upload, configurable non-overlapping splits.
 # This deliberately does not overwrite approved frozen models or policies.
 # -------------------------------------------------------------------------
 adaptive_state <- reactiveVal(NULL)
 adaptive_message <- reactiveVal('No refreshed model yet. Upload a labelled order CSV.')
 output$adaptive_status <- renderText(adaptive_message())
 output$adaptive_test_pct <- renderUI({
   remaining <- 100 - input$adaptive_train - input$adaptive_validation
   div(style='margin-top:26px;',strong(sprintf('Testing: %s%%',remaining)),
       if(remaining < 10 || remaining > 80) span(style='color:#b04532;',
         ' Adjust training/validation so testing is 10–80%.'))
 })
 observeEvent(input$adaptive_go, {
   adaptive_state(NULL)
   if(is.null(input$adaptive_file)) {
     adaptive_message('Select ONE complete order CSV first.');return()
   }
   tr <- input$adaptive_train
   va <- input$adaptive_validation
   te <- 100 - tr - va
   if(te < 10 || te > 80) {
     adaptive_message('Invalid split: testing must be between 10% and 80%.');return()
   }
   if(!nzchar(Sys.which(python)) && !file.exists(python)) {
     adaptive_message('Python not found. Set SUPPLY_PYTHON to the working Python executable.');return()
   }
   folder <- tempfile('adaptive_session_')
   dir.create(folder)
   # Keep output files until the next refresh/session end; Shiny holds their paths.
   tryCatch({
     withProgress(message='Validating data and retraining candidate models',value=.1,{
       args <- c(shQuote(file.path(root,'backend','adaptive_training.py')),
                 shQuote(input$adaptive_file$datapath),shQuote(folder),
                 '--train',as.character(tr),'--validation',as.character(va),
                 '--method',input$adaptive_method)
       msg <- suppressWarnings(system2(python,args,stdout=TRUE,stderr=TRUE))
       code <- attr(msg,'status')
       if(!is.null(code) && code!=0) stop(paste(msg,collapse='\n'))
       incProgress(.8)
       met <- read.csv(file.path(folder,'adaptive_metrics.csv'),check.names=FALSE)
       pr <- readcsv(file.path(folder,'adaptive_predictions.csv'))
       adaptive_state(list(metrics=met,predictions=pr,folder=folder))
       adaptive_message(paste('Model updated in this session.',paste(msg,collapse='\n'),
                              'No GitHub files or historic policy simulations were changed.',sep='\n'))
     })
   },error=function(e){
       adaptive_message(paste('Refresh not completed:',conditionMessage(e)))
       showNotification('Adaptive model refresh failed; inspect the status.',type='error')
   })
 })
 output$adaptive_metrics <- renderDT({
   req(adaptive_state())
   DT::datatable(adaptive_state()$metrics,rownames=FALSE,
                 options=list(pageLength=5,scrollX=TRUE))
 })
 output$adaptive_summary <- renderUI({
   req(adaptive_state())
   p(sprintf('Forecasts refreshed for %s records; %s were open orders.',
      nrow(adaptive_state()$predictions),
      sum(adaptive_state()$predictions$partition=='Open / unlabelled')))
 })
 output$adaptive_predictions <- renderDT({
   req(adaptive_state())
   x <- adaptive_state()$predictions
   x <- x[order(-x$adaptive_late_probability),]
   show <- intersect(c('po_id','item_id','partition','supplier_id','transport_mode',
     'adaptive_predicted_lead_days','adaptive_late_probability','adaptive_lower_90',
     'adaptive_upper_90','risk_priority','available_inventory_days',
     'indicative_next_step'),names(x))
   DT::datatable(x[,show,drop=FALSE],rownames=FALSE,
     options=list(pageLength=10,scrollX=TRUE))
 })
 output$adaptive_download <- downloadHandler(
   filename=function() paste0('adaptive_priorities_',Sys.Date(),'.csv'),
   content=function(file){req(adaptive_state());
     file.copy(file.path(adaptive_state()$folder,'adaptive_predictions.csv'),
               file,overwrite=TRUE)}
 )
 output$adaptive_example <- downloadHandler(
   filename=function() 'All_Orders_example.csv',
   content=function(file) file.copy(file.path(root,'data','All_Orders.csv'),
                                    file,overwrite=TRUE)
 )

 current <- reactiveVal(NULL); status <- reactiveVal('Loading demonstration…'); reviews <- reactiveVal(data.frame(order_key=character(),action=character(),decision=character(),notes=character(),time=character(),stringsAsFactors=FALSE))
 emptylog <- function() reviews(reviews()[0,])
 load_batch <- function(path,label){
  current(NULL);emptylog();status('Validating and scoring batch…')
  td <- tempfile('command_');dir.create(td);on.exit(unlink(td,recursive=TRUE),add=TRUE)
  tryCatch({if(!file.exists(path)) stop('Input file missing. Extract the complete application folder.');if(!nzchar(Sys.which(python))&&!file.exists(python))stop('Python interpreter not found. Configure SUPPLY_PYTHON using the installation guide.');withProgress(message='Generating forecasts and policy scenarios',value=.2,{
   msg<-suppressWarnings(system2(python,c(shQuote(file.path(root,'backend','command.py')),shQuote(path),shQuote(td)),stdout=TRUE,stderr=TRUE));code<-attr(msg,'status');if(!is.null(code)&&code!=0)stop(paste(msg,collapse='\n'))
   incProgress(.7); current(list(d=readcsv(file.path(td,'results.csv')),s=readcsv(file.path(td,'scenarios.csv')),q=read.csv(file.path(td,'quality.csv')),label=label));status(paste(label,paste(msg,collapse='\n'),sep='\n'))
  })},error=function(e){status(paste('Batch rejected:',conditionMessage(e)));showNotification('Batch rejected; see status.',type='error')})
 }
 demo <- function(){tryCatch({current(list(d=readcsv(file.path(root,'tests','demo','results.csv')),s=readcsv(file.path(root,'tests','demo','scenarios.csv')),q=read.csv(file.path(root,'tests','demo','quality.csv')),label='Synthetic demonstration'));emptylog();status('Verified demonstration batch: 1,000 orders.')},error=function(e){current(NULL);emptylog();status('Demonstration files unavailable. Extract the complete application folder.')})}
 observeEvent(TRUE,demo(),once=TRUE);observeEvent(input$demo,demo());observeEvent(input$run,{if(is.null(input$upload)){current(NULL);emptylog();status('Choose an order CSV using Browse first.');return()};load_batch(input$upload$datapath,input$upload$name)})
 output$status<-renderText(status())
 observeEvent(input$reset_filters,{updateSelectInput(session,'plant',selected='All');updateSelectInput(session,'transport',selected='All');updateSelectInput(session,'critical',selected='All');updateSliderInput(session,'amber',value=.4);updateSliderInput(session,'red',value=.75);updateSelectInput(session,'level',selected='90');updateSelectInput(session,'prob',selected='raw_late_probability')})
 observeEvent(current(),{x<-current()$d;updateSelectInput(session,'plant',choices=c('All',sort(unique(x$plant_id))),selected='All')})
 filtered<-reactive({req(current());validate(need(input$amber<input$red,'Review threshold must be below high-risk threshold.'));x<-current()$d;if(!is.null(input$plant)&&input$plant!='All')x<-x[x$plant_id==input$plant,];if(input$transport!='All')x<-x[x$transport_mode==input$transport,];if(input$critical!='All')x<-x[x$material_criticality==input$critical,];x$delay_probability<-x[[input$prob]];x$alert<-ifelse(x$delay_probability>=input$red,'High',ifelse(x$delay_probability>=input$amber,'Review','Low'));x$inventory_days<-x$inventory_on_hand/x$daily_demand;x$lower<-x[[paste0('group_lower_',input$level)]];x$upper<-x[[paste0('group_upper_',input$level)]];validate(need(nrow(x)>0,'No matching orders. Reset the filters.'));x[order(-x$delay_probability,-x$estimated_savings),]})
 observeEvent(filtered(),{x<-filtered();updateSelectInput(session,'order',choices=key(x));updateSelectInput(session,'review_order',choices=key(x))})
 dt<-function(x)datatable(x,rownames=FALSE,escape=TRUE,options=list(pageLength=10,scrollX=TRUE))
 cols<-c('po_id','item_id','supplier_id','plant_id','transport_mode','material_criticality','planned_lead_days','predicted_lead_days','lower','upper','delay_probability','alert','anomaly_flag','unseen_category','action')
 output$kpis<-renderUI({x<-filtered();div(class='kpis',kpi('Orders',nrow(x)),kpi('High-risk orders',sum(x$alert=='High')),kpi('Anomalies',sum(x$anomaly_flag)),kpi('Mean forecast',paste(fmt(mean(x$predicted_lead_days)),'days')),kpi('Potential stockout exposure',pct(mean(x$stockout_probability))),kpi('Estimated savings / order',euro(mean(x$estimated_savings))))})
 output$riskplot<-renderPlot({ggplot(filtered(),aes(factor(alert,levels=c('Low','Review','High')),fill=alert))+geom_bar()+scale_fill_manual(values=c(Low='#37856e',Review='#d49a46',High='#ba4b4b'))+labs(x=NULL,y='Orders')+theme_minimal()+theme(legend.position='none')})
 output$actions<-renderPlot({ggplot(filtered(),aes(action))+geom_bar(fill='#176e76')+coord_flip()+labs(x=NULL,y='Orders')+theme_minimal()})
 output$priority<-renderDT(dt(head(filtered()[,cols],25)))
 output$alerts_table<-renderDT({x<-filtered();if(input$alerts)x<-x[x$alert!='Low'|x$anomaly_flag==1,];dt(x[,cols])})
 output$intervals<-renderPlot({x<-head(filtered(),40);validate(need(nrow(x)>0,'No matching orders.'));x$i<-seq_len(nrow(x));ggplot(x,aes(i,predicted_lead_days))+geom_linerange(aes(ymin=lower,ymax=upper),colour='#8aa6bc')+geom_point(colour='#176e76')+geom_point(aes(y=planned_lead_days),colour='#bc8c46',shape=4)+labs(x='Top risk orders',y='Days',subtitle='Cross = planned; dot = predicted')+theme_minimal()})
 output$suppliers<-renderDT({x<-filtered();validate(need(nrow(x)>0,'No matching orders.'));z<-aggregate(cbind(delay_probability,predicted_lead_days,anomaly_flag,stockout_probability)~supplier_id,data=x,FUN=mean);n<-as.data.frame(table(x$supplier_id));names(n)<-c('supplier_id','orders');z<-merge(z,n);cl<-unique(x[,c('supplier_id','supplier_cluster')]);z<-merge(z,cl,all.x=TRUE);dt(z[order(-z$delay_probability),])})
 output$anomalies<-renderDT({x<-filtered();x<-x[x$anomaly_flag==1,];dt(x[order(-x$anomaly_score),c(cols,'anomaly_score')])})
 output$inventory_kpis<-renderUI({x<-filtered();div(class='kpis',kpi('Additional buffer units',fmt(sum(x$buffer_units),0)),kpi('Alternate sourcing orders',sum(grepl('Alternate',x$action))),kpi('Expedite orders',sum(grepl('Expedite',x$action))),kpi('Expected shortage units / order',fmt(mean(x$shortage_units))))})
 output$decisions<-renderDT({x<-filtered();dt(x[,c('po_id','item_id','supplier_id','action','inventory_days','buffer_units','alternate_supplier_id','alternate_capacity','order_qty','baseline_cost','cost','estimated_savings','stockout_probability')])})
 output$detail<-renderUI({x<-filtered();x<-x[key(x)==input$order,];req(nrow(x)==1);tagList(h4(x$action),p(paste('Lead-time forecast:',fmt(x$predicted_lead_days),'days; interval',fmt(x$lower),'to',fmt(x$upper),'days.')),p(paste('Suggested buffer:',fmt(x$buffer_units,0),'units; inventory coverage:',fmt(x$inventory_days),'days.')),p(paste('Alternative supplier:',x$alternate_supplier_id,'; capacity:',x$alternate_capacity,'; order qty:',x$order_qty)),p(paste('Expected cost:',euro(x$cost),'; estimated savings:',euro(x$estimated_savings))),p(paste('Potential stockout risk before/after:',pct(x$baseline_stockout_probability),'/',pct(x$stockout_probability))))})
 scenario<-reactive({x<-filtered();s<-current()$s;s<-s[key(s)%in%key(x)&s$stress==as.integer(input$stress),];validate(need(nrow(s)>0,'No matching orders.'));aggregate(cbind(cost,stockout_probability,shortage_units,lead_days,buffer_units)~policy,data=s,FUN=mean)})
 output$scenario_kpis<-renderUI({s<-scenario();b<-s$cost[s$policy=='Maintain'];o<-s$cost[s$policy=='Optimised'];div(class='kpis',kpi('Baseline cost / order',euro(b)),kpi('Optimised cost / order',euro(o)),kpi('Estimated reduction',pct(if(b>0)1-o/b else NA_real_)),kpi('Optimised stockout exposure',pct(s$stockout_probability[s$policy=='Optimised'])))})
 output$costplot<-renderPlot({ggplot(scenario(),aes(reorder(policy,cost),cost))+geom_col(fill='#176e76')+coord_flip()+labs(x=NULL,y='Expected cost (EUR/order)')+theme_minimal()})
 output$stockplot<-renderPlot({ggplot(scenario(),aes(reorder(policy,stockout_probability),100*stockout_probability))+geom_col(fill='#c38a42')+coord_flip()+labs(x=NULL,y='Potential stockout exposure (%)')+theme_minimal()})
 output$scenario_table<-renderDT(dt(scenario()))
 observeEvent(input$save_review,{x<-filtered();x<-x[key(x)==input$review_order,];req(nrow(x)==1);reviews(rbind(reviews(),data.frame(order_key=key(x),action=x$action,decision=input$review_status,notes=input$review_note,time=format(Sys.time(),tz='UTC',usetz=TRUE),stringsAsFactors=FALSE)));showNotification('Review recorded for this session.')})
 output$review_log<-renderDT(dt(reviews()))
 dl<-function(name,f)downloadHandler(filename=function()paste0(name,'_',Sys.Date(),'.csv'),content=function(file)write.csv(f(),file,row.names=FALSE,na=''))
 output$export<-dl('command_orders',filtered);output$scenario_export<-dl('scenario_comparison',scenario);output$review_export<-dl('planner_review',reviews);output$quality_export<-dl('validation',function(){req(current());current()$q});output$upload_example<-downloadHandler(filename=function() 'Upload_Example.csv',content=function(file)file.copy(file.path(root,'data','Upload_Example.csv'),file,overwrite=TRUE));output$schema<-dl('data_dictionary',function()read.csv(file.path(root,'data','Data_Dictionary.csv'),check.names=FALSE))
 output$plant_risk<-renderPlot({x<-filtered();validate(need(nrow(x)>0,'No matching orders.'));ggplot(x,aes(plant_id,fill=alert))+geom_bar(position='fill')+labs(x='Plant',y='Risk fraction')+theme_minimal()})
 output$mode_risk<-renderPlot({x<-filtered();validate(need(nrow(x)>0,'No matching orders.'));ggplot(x,aes(transport_mode,delay_probability))+geom_boxplot(fill='#b3d9d5')+labs(x='Transport',y='Delay probability')+theme_minimal()})
 output$supplier_risk<-renderPlot({x<-filtered();validate(need(nrow(x)>0,'No matching orders.'));z<-aggregate(delay_probability~supplier_id,x,mean);z<-head(z[order(-z$delay_probability),],15);ggplot(z,aes(reorder(supplier_id,delay_probability),delay_probability))+geom_col(fill='#147f79')+coord_flip()+labs(x=NULL,y='Mean delay probability (top 15)')+theme_minimal()})
 output$anomaly_hist<-renderPlot({ggplot(filtered(),aes(anomaly_score))+geom_histogram(bins=30,fill='#3979a8')+geom_vline(xintercept=0,linetype=2)+labs(x='Anomaly score',y='Orders')+theme_minimal()})
 output$savings_plot<-renderPlot({ggplot(filtered(),aes(action,estimated_savings))+geom_boxplot(fill='#b3d9d5')+coord_flip()+labs(x=NULL,y='Estimated EUR saved/order')+theme_minimal()})
 output$inventory_plot<-renderPlot({x<-filtered();ggplot(x,aes(inventory_days,upper,colour=alert))+geom_point(alpha=.5)+geom_abline(linetype=2)+labs(x='Available inventory (days)',y='Upper lead-time limit (days)')+theme_minimal()})
 output$stress_trend<-renderPlot({x<-filtered();z<-current()$s;z<-z[key(z)%in%key(x),];validate(need(nrow(z)>0,'No matching orders.'));z<-aggregate(cost~stress+policy,z,mean);ggplot(z,aes(stress,cost,colour=policy))+geom_line()+geom_point()+scale_x_continuous(breaks=0:2,labels=c('Base','Moderate','Severe'))+labs(x=NULL,y='Expected EUR/order')+theme_minimal()})
 observeEvent(input$inventory_click,{x<-filtered();q<-nearPoints(x,input$inventory_click,xvar='inventory_days',yvar='upper',threshold=15,maxpoints=1);if(nrow(q))updateSelectInput(session,'order',selected=key(q))})
}
shinyApp(ui,server)
