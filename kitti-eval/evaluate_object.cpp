#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <strings.h>
#include <assert.h>
#include <string>

#include <dirent.h>

#include <fstream>
#include <sstream>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>

BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)

typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > Polygon;


using namespace std;

/*=======================================================================
STATIC EVALUATION PARAMETERS
=======================================================================*/

// holds the number of test images on the server
//const int32_t N_TESTIMAGES = 7518;
int32_t N_TESTIMAGES = 7480;

const int32_t append_zeros = 6;

// easy, moderate and hard evaluation level
enum DIFFICULTY{EASY=0, MODERATE=1, HARD=2, ALL=3};

// evaluation metrics: image, ground or 3D
enum METRIC{IMAGE=0, GROUND=1, BOX3D=2};

// evaluation parameter
const int32_t MIN_HEIGHT[4]     = {40, 25, 25, 0};      // minimum height for evaluated groundtruth/detections
const int32_t MAX_OCCLUSION[4]  = {0, 1, 2, 2};         // maximum occlusion level of the groundtruth used for evaluation
const double  MAX_TRUNCATION[4] = {0.15, 0.3, 0.5, 1.}; // maximum truncation level of the groundtruth used for evaluation
const int32_t MAX_Z[4]   = {-1, -1, -1, -1};            // maximum distance from the ego-car

// evaluated object classes
enum CLASSES{PEDESTRIAN=1};
const int NUM_CLASS = 1;

// parameters varying per class
vector<string> CLASS_NAMES;
// the minimum overlap required for 2D evaluation on the image/ground plane and 3D evaluation
const double   MIN_OVERLAP[3][NUM_CLASS] = {{0.3}, {0.5}, {0.5}};
// maximum relative error for detection
const int NUM_RELATIVE_ERROR = 3;
const double   MAX_RELATIVE_ERROR[NUM_RELATIVE_ERROR][NUM_CLASS] = {{0.01}, {0.05}, {0.10}};

// no. of recall steps that should be evaluated (discretized)
const double N_SAMPLE_PTS = 41;
const int N_IOU_SAMPLE_PTS = 51;

const int VIEWP_BINS = 8;
const double VIEWP_OFFSET = 0.3927;

const int MIN_DIST = 10;
const int DELTA_DIST = 5;
const int MAX_DIST = 60;

const double MIN_SCORE = -1000.0;

// initialize class names
void initGlobals () {
  CLASS_NAMES.push_back("pedestrian");
}

/*=======================================================================
DATA TYPES FOR EVALUATION
=======================================================================*/

// holding data needed for precision-recall and precision-aos
struct tPrData {
  vector<double> v;           // detection score for computing score thresholds
  double         similarity;  // orientation similarity
  int32_t        tp;          // true positives
  int32_t        fp;          // false positives
  int32_t        fn;          // false negatives+
  vector<int32_t> pred_bins;
  vector<int32_t> tp_bins;
  tPrData () :
    similarity(0), tp(0), fp(0), fn(0) {
      pred_bins.assign(VIEWP_BINS, 0);
      tp_bins.assign(VIEWP_BINS, 0);
    }
};

// holding bounding boxes for ground truth and detections
struct tBox {
  string  type;     // object type as car, pedestrian or cyclist,...
  double   x1;      // left corner
  double   y1;      // top corner
  double   x2;      // right corner
  double   y2;      // bottom corner
  double   alpha;   // image orientation
  tBox (string type, double x1,double y1,double x2,double y2,double alpha) :
    type(type),x1(x1),y1(y1),x2(x2),y2(y2),alpha(alpha) {}
};

// holding ground truth data
struct tGroundtruth {
  tBox    box;        // object type, box, orientation
  double  truncation; // truncation 0..1
  int32_t occlusion;  // occlusion 0,1,2 (non, partly, fully)
  double ry;
  double  t1, t2, t3;
  double h, w, l;
  tGroundtruth () :
    box(tBox("invalild",-1,-1,-1,-1,-10)),truncation(-1),occlusion(-1) {}
  tGroundtruth (tBox box,double truncation,int32_t occlusion) :
    box(box),truncation(truncation),occlusion(occlusion) {}
  tGroundtruth (string type,double x1,double y1,double x2,double y2,double alpha,double truncation,int32_t occlusion) :
    box(tBox(type,x1,y1,x2,y2,alpha)),truncation(truncation),occlusion(occlusion) {}
};

// holding detection data
struct tDetection {
  tBox    box;    // object type, box, orientation
  double  thresh; // detection score
  double  ry;
  double  t1, t2, t3;
  double  h, w, l;
  tDetection ():
    box(tBox("invalid",-1,-1,-1,-1,-10)),thresh(-1000) {}
  tDetection (tBox box,double thresh) :
    box(box),thresh(thresh) {}
  tDetection (string type,double x1,double y1,double x2,double y2,double alpha,double thresh) :
    box(tBox(type,x1,y1,x2,y2,alpha)),thresh(thresh) {}
};


/*=======================================================================
FUNCTIONS TO LOAD DETECTION AND GROUND TRUTH DATA ONCE, SAVE RESULTS
=======================================================================*/
vector<tDetection> loadDetections(string file_name, bool &compute_aos,
        vector<bool> &eval_image, vector<bool> &eval_ground,
        vector<bool> &eval_3d, bool &success, vector<int> &count) {

  // holds all detections (ignored detections are indicated by an index vector
  vector<tDetection> detections;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return detections;
  }
  while (!feof(fp)) {
    tDetection d;
    double trash;
    char str[255];
    double score;
    if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &trash, &trash, &d.box.alpha, &d.box.x1, &d.box.y1,
                   &d.box.x2, &d.box.y2, &d.h, &d.w, &d.l, &d.t1, &d.t2, &d.t3,
                   &d.ry, &score)==16) {

      d.box.type = str;
      d.thresh = score;
      //d.thresh = exp(score);
      if (d.thresh < MIN_SCORE)
        continue;
      detections.push_back(d);

      // orientation=-10 is invalid, AOS is not evaluated if at least one orientation is invalid
      if(d.box.alpha == -10)
        compute_aos = false;

      // a class is only evaluated if it is detected at least once
      for (int c = 0; c < CLASS_NAMES.size(); c++){
        if (!strcasecmp(d.box.type.c_str(), CLASS_NAMES[c].c_str())){
          count[c]++;
          if (!eval_image[c] && d.box.x1 >= 0)
            eval_image[c] = true;
          if (!eval_ground[c] && d.t1 != -1000 && d.t3 != -1000 && d.w > 0 && d.l > 0)
            eval_ground[c] = true;
          if (!eval_3d[c] && d.t1 != -1000 && d.t2 != -1000 && d.t3 != -1000 && d.h > 0 && d.w > 0 && d.l > 0)
            eval_3d[c] = true;
          break;
        }
      }
    }
  }

  fclose(fp);
  success = true;
  return detections;
}

vector<tGroundtruth> loadGroundtruth(string file_name,bool &success, vector<int> &count) {

  // holds all ground truth (ignored ground truth is indicated by an index vector
  vector<tGroundtruth> groundtruth;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return groundtruth;
  }
  while (!feof(fp)) {
    tGroundtruth g;
    char str[255];
    if (fscanf(fp, "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str,         &g.truncation, &g.occlusion, &g.box.alpha,
                   &g.box.x1,   &g.box.y1,     &g.box.x2,    &g.box.y2,
                   &g.h,        &g.w,          &g.l,         &g.t1,
                   &g.t2,       &g.t3,         &g.ry)==15){
      g.box.type = str;
      groundtruth.push_back(g);
    }
    for (int cls_idx=0; cls_idx<CLASS_NAMES.size(); cls_idx++){
      CLASSES cls = static_cast<CLASSES>(cls_idx);
      if(!strcasecmp(g.box.type.c_str(), CLASS_NAMES[cls].c_str())){
        count[cls_idx]++;
        break;
      }
    }
  }
  fclose(fp);
  success = true;
  return groundtruth;
}

void saveStats (const vector<double> &precision, const vector<double> &aos, const vector<double> &recalls, const vector<double> &mppe, FILE *fp_det, FILE *fp_ori, FILE *fp_iou, FILE *fp_mppe) {

  // save precision to file if requested
  if(precision.empty())
    return;

  if(fp_det!=NULL){
    for (int32_t i=0; i<precision.size(); i++)
      fprintf(fp_det,"%f ",precision[i]);
    fprintf(fp_det,"\n");
  }

  // save orientation similarity, only if there were no invalid orientation entries in submission (alpha=-10)
  if(!aos.empty() && fp_ori!=NULL){
    for (int32_t i=0; i<aos.size(); i++)
      fprintf(fp_ori,"%f ",aos[i]);
    fprintf(fp_ori,"\n");
  }

  // save recall vs IOU if requested
  if(!recalls.empty() && fp_iou!=NULL){
    for (int32_t i=0; i<recalls.size(); i++)
      fprintf(fp_iou,"%f ",recalls[i]);
    fprintf(fp_iou,"\n");
  }

  // save MPPE vs recall if requested
  if(!mppe.empty() && fp_mppe!=NULL){
    for (int32_t i=0; i<mppe.size(); i++)
      fprintf(fp_mppe,"%f ",mppe[i]);
    fprintf(fp_mppe,"\n");
  }

}

/*=======================================================================
EVALUATION HELPER FUNCTIONS
=======================================================================*/

// criterion defines whether the overlap is computed with respect to both areas (ground truth and detection)
// or with respect to box a or b (detection and "dontcare" areas)
inline double imageBoxOverlap(tBox a, tBox b, int32_t criterion=-1){

  // overlap is invalid in the beginning
  double o = -1;

  // get overlapping area
  double x1 = max(a.x1, b.x1);
  double y1 = max(a.y1, b.y1);
  double x2 = min(a.x2, b.x2);
  double y2 = min(a.y2, b.y2);

  // compute width and height of overlapping area
  double w = x2-x1;
  double h = y2-y1;

  // set invalid entries to 0 overlap
  if(w<=0 || h<=0)
    return 0;

  // get overlapping areas
  double inter = w*h;
  double a_area = (a.x2-a.x1) * (a.y2-a.y1);
  double b_area = (b.x2-b.x1) * (b.y2-b.y1);

  // intersection over union overlap depending on users choice
  if(criterion==-1)     // union
    o = inter / (a_area+b_area-inter);
  else if(criterion==0) // bbox_a
    o = inter / a_area;
  else if(criterion==1) // bbox_b
    o = inter / b_area;

  // overlap
  return o;
}

inline double imageBoxOverlap(tDetection a, tGroundtruth b, int32_t criterion=-1, double relative_error=0.){
  return imageBoxOverlap(a.box, b.box, criterion);
}

// compute image box overlap only if distance is lower than a relative error
inline double imageBoxOverlapWithRelativeError(tDetection d, tGroundtruth g, int32_t criterion=-1, double relative_error=0.) {
  double box_overlap = imageBoxOverlap(d.box, g.box, criterion);
  double distance = sqrt(pow(g.t1-d.t1,2) + pow(g.t2-d.t2,2) + pow(g.t3-d.t3,2));
  double threshold = relative_error * sqrt(pow(g.t1,2) + pow(g.t2,2) + pow(g.t3,2)) + 0.20; // with a margin of 20cm
  if (distance <= threshold)
    return box_overlap;
  else
    return 0.;
}

// compute polygon of an oriented bounding box
template <typename T>
Polygon toPolygon(const T& g) {
    using namespace boost::numeric::ublas;
    using namespace boost::geometry;
    matrix<double> mref(2, 2);
    mref(0, 0) = cos(g.ry); mref(0, 1) = sin(g.ry);
    mref(1, 0) = -sin(g.ry); mref(1, 1) = cos(g.ry);

    static int count = 0;
    matrix<double> corners(2, 4);
    double data[] = {g.l / 2, g.l / 2, -g.l / 2, -g.l / 2,
                     g.w / 2, -g.w / 2, -g.w / 2, g.w / 2};
    std::copy(data, data + 8, corners.data().begin());
    matrix<double> gc = prod(mref, corners);
    for (int i = 0; i < 4; ++i) {
        gc(0, i) += g.t1;
        gc(1, i) += g.t3;
    }

    double points[][2] = {{gc(0, 0), gc(1, 0)},{gc(0, 1), gc(1, 1)},{gc(0, 2), gc(1, 2)},{gc(0, 3), gc(1, 3)},{gc(0, 0), gc(1, 0)}};
    Polygon poly;
    append(poly, points);
    return poly;
}

// measure overlap between bird's eye view bounding boxes, parametrized by (ry, l, w, tx, tz)
inline double groundBoxOverlap(tDetection d, tGroundtruth g, int32_t criterion = -1, double relative_error=0.) {
    using namespace boost::geometry;
    Polygon gp = toPolygon(g);
    Polygon dp = toPolygon(d);

    std::vector<Polygon> in, un;
    intersection(gp, dp, in);
    union_(gp, dp, un);

    double inter_area = in.empty() ? 0 : area(in.front());
    double union_area = area(un.front());
    double o;
    if(criterion==-1)     // union
        o = inter_area / union_area;
    else if(criterion==0) // bbox_a
        o = inter_area / area(dp);
    else if(criterion==1) // bbox_b
        o = inter_area / area(gp);

    return o;
}

// measure overlap between 3D bounding boxes, parametrized by (ry, h, w, l, tx, ty, tz)
inline double box3DOverlap(tDetection d, tGroundtruth g, int32_t criterion = -1, double relative_error=0.) {
    using namespace boost::geometry;
    Polygon gp = toPolygon(g);
    Polygon dp = toPolygon(d);

    std::vector<Polygon> in, un;
    intersection(gp, dp, in);
    union_(gp, dp, un);

    double ymax = min(d.t2, g.t2);
    double ymin = max(d.t2 - d.h, g.t2 - g.h);

    double inter_area = in.empty() ? 0 : area(in.front());
    double inter_vol = inter_area * max(0.0, ymax - ymin);

    double det_vol = d.h * d.l * d.w;
    double gt_vol = g.h * g.l * g.w;

    double o;
    if(criterion==-1)     // union
        o = inter_vol / (det_vol + gt_vol - inter_vol);
    else if(criterion==0) // bbox_a
        o = inter_vol / det_vol;
    else if(criterion==1) // bbox_b
        o = inter_vol / gt_vol;

    return o;
}

vector<double> getThresholds(vector<double> &v, double n_groundtruth){

  // holds scores needed to compute N_SAMPLE_PTS recall values
  vector<double> t;

  // sort scores in descending order
  // (highest score is assumed to give best/most confident detections)
  sort(v.begin(), v.end(), greater<double>());

  // get scores for linearly spaced recall
  double current_recall = 0;
  for(int32_t i=0; i<v.size(); i++){

    // check if right-hand-side recall with respect to current recall is close than left-hand-side one
    // in this case, skip the current detection score
    double l_recall, r_recall, recall;
    l_recall = (double)(i+1)/n_groundtruth;
    if(i<(v.size()-1))
      r_recall = (double)(i+2)/n_groundtruth;
    else
      r_recall = l_recall;

    if( (r_recall-current_recall) < (current_recall-l_recall) && i<(v.size()-1))
      continue;

    // left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    // the next recall step was reached
    t.push_back(v[i]);
    current_recall += 1.0/(N_SAMPLE_PTS-1.0);
  }
  return t;
}

void cleanData(CLASSES current_class, const vector<tGroundtruth> &gt, const vector<tDetection> &det, vector<int32_t> &ignored_gt, vector<tGroundtruth> &dc, vector<int32_t> &ignored_det, int32_t &n_gt, DIFFICULTY difficulty, int fixed_max_z=-1){

  // select max distance from ego-vehicle from either function parameter or global variable
  int max_z = fixed_max_z>0 ? fixed_max_z : MAX_Z[difficulty];

  // extract ground truth bounding boxes for current evaluation class
  for(int32_t i=0;i<gt.size(); i++){

    // only bounding boxes with a minimum height are used for evaluation
    double height = gt[i].box.y2 - gt[i].box.y1;

    // neighboring classes are ignored ("van" for "car" and "person_sitting" for "pedestrian")
    // (lower/upper cases are ignored)
    int32_t valid_class;

    // all classes without a neighboring class
    if(!strcasecmp(gt[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;

    // classes with a neighboring class
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "Pedestrian") && !strcasecmp("Person_sitting", gt[i].box.type.c_str()))
      valid_class = 0;
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "Car") && !strcasecmp("Van", gt[i].box.type.c_str()))
      valid_class = 0;
    else if (!strcasecmp(CLASS_NAMES[current_class].c_str(), "Person_sitting") && !strcasecmp("Pedestrian", gt[i].box.type.c_str()))
      valid_class = 0;
    else if (!strcasecmp(CLASS_NAMES[current_class].c_str(), "Van") && !strcasecmp("Car", gt[i].box.type.c_str()))
      valid_class = 0;
    // classes not used for evaluation
    else
      valid_class = -1;

    // ground truth is ignored, if occlusion, truncation exceeds the difficulty or ground truth is too small
    // (doesn't count as FN nor TP, although detections may be assigned)
    bool ignore = false;
    double distance = sqrt(pow(gt[i].t3,2)+pow(gt[i].t1,2));
    int cat = -1;

    if(gt[i].occlusion <= MAX_OCCLUSION[0] && gt[i].truncation <= MAX_TRUNCATION[0] && height > MIN_HEIGHT[0])
    {cat = 0;}

    else if(gt[i].occlusion <= MAX_OCCLUSION[1] && gt[i].truncation <= MAX_TRUNCATION[1] && height > MIN_HEIGHT[1])
    {cat = 1;}

    else if(gt[i].occlusion <= MAX_OCCLUSION[2] && gt[i].truncation <= MAX_TRUNCATION[2] && height > MIN_HEIGHT[2])
    {cat = 2;}

    if ((difficulty != ALL) && (difficulty != cat)){ignore=true;}

    // set ignored vector for ground truth
    // current class and not ignored (total no. of ground truth is detected for recall denominator)
    if(valid_class==1 && !ignore){
      ignored_gt.push_back(0);
      n_gt++;
    }

    // neighboring class, or current class but ignored
    else if(valid_class==0 || (ignore && valid_class==1))
      ignored_gt.push_back(1);

    // all other classes which are FN in the evaluation
    else
      ignored_gt.push_back(-1);
  }

  // extract dontcare areas
  for(int32_t i=0;i<gt.size(); i++)
    if(!strcasecmp("DontCare", gt[i].box.type.c_str()))
      dc.push_back(gt[i]);

  // extract detections bounding boxes of the current class
  for(int32_t i=0;i<det.size(); i++){

    // neighboring classes are not evaluated
    int32_t valid_class;
    if(!strcasecmp(det[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;
    else
      valid_class = -1;

    int32_t height = fabs(det[i].box.y1 - det[i].box.y2);

    // set ignored vector for detections
    if(height<MIN_HEIGHT[difficulty])
      ignored_det.push_back(1);
    else if(valid_class==1)
      ignored_det.push_back(0);
    else
      ignored_det.push_back(-1);
  }
}

tPrData computeStatistics(CLASSES current_class, const vector<tGroundtruth> &gt,
                          const vector<tDetection> &det, const vector<tGroundtruth> &dc,
                          const vector<int32_t> &ignored_gt, const vector<int32_t>  &ignored_det,
                          bool compute_fp, double (*boxoverlap)(tDetection, tGroundtruth, int32_t, double),
                          METRIC metric, bool compute_aos=false, double thresh=0, double fixed_iou=-1,
                          double relative_error=0.){

  tPrData stat = tPrData();
  const double NO_DETECTION = -10000000;
  vector<double> delta;            // holds angular difference for TPs (needed for AOS evaluation)
  vector<int> pred_bin;
  pred_bin.assign(VIEWP_BINS, 0);
  vector<int> tp_bin;
  tp_bin.assign(VIEWP_BINS, 0);
  vector<bool> assigned_detection; // holds wether a detection was assigned to a valid or ignored ground truth
  assigned_detection.assign(det.size(), false);
  vector<bool> ignored_threshold;
  ignored_threshold.assign(det.size(), false); // holds detections with a threshold lower than thresh if FP are computed

  // select min IOU from either parameter or global variable
  double min_overlap = fixed_iou>0 ? fixed_iou : MIN_OVERLAP[metric][current_class];

  // detections with a low score are ignored for computing precision (needs FP)
  if(compute_fp)
    for(int32_t i=0; i<det.size(); i++)
      if(det[i].thresh<thresh)
        ignored_threshold[i] = true;

  // evaluate all ground truth boxes
  for(int32_t i=0; i<gt.size(); i++){

    // this ground truth is not of the current or a neighboring class and therefore ignored
    if(ignored_gt[i]==-1)
      continue;

    /*=======================================================================
    find candidates (overlap with ground truth > 0.5) (logical len(det))
    =======================================================================*/
    int32_t det_idx          = -1;
    double valid_detection = NO_DETECTION;
    double max_overlap     = 0;

    // search for a possible detection
    bool assigned_ignored_det = false;
    for(int32_t j=0; j<det.size(); j++){

      // detections not of the current class, already assigned or with a low threshold are ignored
      if(ignored_det[j]==-1)
        continue;
      if(assigned_detection[j])
        continue;
      if(ignored_threshold[j])
        continue;

      // find the maximum score for the candidates and get idx of respective detection
      double overlap = boxoverlap(det[j], gt[i], -1, relative_error);

      // for computing recall thresholds, the candidate with highest score is considered
      if(!compute_fp && overlap>min_overlap && det[j].thresh>valid_detection){
        det_idx         = j;
        valid_detection = det[j].thresh;
      }

      // for computing pr curve values, the candidate with the greatest overlap is considered
      // if the greatest overlap is an ignored detection (min_height), the overlapping detection is used
      else if(compute_fp && overlap>min_overlap && (overlap>max_overlap || assigned_ignored_det) && ignored_det[j]==0){
        max_overlap     = overlap;
        det_idx         = j;
        valid_detection = 1;
        assigned_ignored_det = false;
      }
      else if(compute_fp && overlap>min_overlap && valid_detection==NO_DETECTION && ignored_det[j]==1){
        det_idx              = j;
        valid_detection      = 1;
        assigned_ignored_det = true;
      }
    }

    /*=======================================================================
    compute TP, FP and FN
    =======================================================================*/

    // nothing was assigned to this valid ground truth
    if(valid_detection==NO_DETECTION && ignored_gt[i]==0) {
      stat.fn++;
    }

    // only evaluate valid ground truth <=> detection assignments (considering difficulty level)
    else if(valid_detection!=NO_DETECTION && (ignored_gt[i]==1 || ignored_det[det_idx]==1))
      assigned_detection[det_idx] = true;

    // found a valid true positive
    else if(valid_detection!=NO_DETECTION){

      // write highest score to threshold vector
      stat.tp++;
      stat.v.push_back(det[det_idx].thresh);

      // compute angular difference of detection and ground truth if valid detection orientation was provided
      if(compute_aos){
        delta.push_back(gt[i].box.alpha - det[det_idx].box.alpha);

        // MPPE computation
        double positive_gt_angle = gt[i].box.alpha;
        positive_gt_angle += positive_gt_angle<0 ? 2*M_PI : 0;
        int gt_bin = std::floor((positive_gt_angle+VIEWP_OFFSET)/(2*M_PI/VIEWP_BINS));

        double positive_det_angle = det[det_idx].box.alpha;
        positive_det_angle += positive_det_angle<0 ? 2*M_PI : 0;
        int det_bin = std::floor((positive_det_angle+VIEWP_OFFSET)/(2*M_PI/VIEWP_BINS));

        if (det_bin>=VIEWP_BINS)
            det_bin = 0;

        if (gt_bin>=VIEWP_BINS)
            gt_bin = 0;

        assert(det_bin<VIEWP_BINS && det_bin>=0);
        assert(gt_bin<VIEWP_BINS && gt_bin>=0);

        pred_bin[det_bin]++;
        tp_bin[det_bin] += (gt_bin == det_bin) ? 1 : 0;
      }

      // clean up
      assigned_detection[det_idx] = true;
    }
  }

  // if FP are requested, consider stuff area
  if(compute_fp){

    // count fp
    for(int32_t i=0; i<det.size(); i++){

      // count false positives if required (height smaller than required is ignored (ignored_det==1)
      if(!(assigned_detection[i] || ignored_det[i]==-1 || ignored_det[i]==1 || ignored_threshold[i]))
        stat.fp++;
    }

    // do not consider detections overlapping with stuff area
    int32_t nstuff = 0;
    for(int32_t i=0; i<dc.size(); i++){
      for(int32_t j=0; j<det.size(); j++){

        // detections not of the current class, already assigned, with a low threshold or a low minimum height are ignored
        if(assigned_detection[j])
          continue;
        if(ignored_det[j]==-1 || ignored_det[j]==1)
          continue;
        if(ignored_threshold[j])
          continue;

        // compute overlap and assign to stuff area, if overlap exceeds class specific value
        double overlap = boxoverlap(det[j], dc[i], 0, relative_error);
        if(overlap>min_overlap){
          assigned_detection[j] = true;
          nstuff++;
        }
      }
    }

    // FP = no. of all not to ground truth assigned detections - detections assigned to stuff areas
    stat.fp -= nstuff;

    // if all orientation values are valid, the AOS is computed
    if(compute_aos){
      vector<double> tmp;

      // FP have a similarity of 0, for all TP compute AOS
      tmp.assign(stat.fp, 0);
      for(int32_t i=0; i<delta.size(); i++)
        tmp.push_back((1.0+cos(delta[i]))/2.0);

      // be sure, that all orientation deltas are computed
      assert(tmp.size()==stat.fp+stat.tp);
      assert(delta.size()==stat.tp);

      // get the mean orientation similarity for this image
      if(stat.tp>0 || stat.fp>0){
        stat.similarity = accumulate(tmp.begin(), tmp.end(), 0.0);
        for (int vp=0; vp<VIEWP_BINS; vp++){
          stat.tp_bins[vp] = tp_bin[vp];
          stat.pred_bins[vp] = pred_bin[vp];
        }
      // there was neither a FP nor a TP, so the similarity is ignored in the evaluation
      }else{
        stat.similarity = -1;
        for (int vp=0; vp<VIEWP_BINS; vp++){
          stat.tp_bins[vp] = -1;
          stat.pred_bins[vp] = -1;
        }
      }
    }
  }
  return stat;
}

/*=======================================================================
EVALUATE CLASS-WISE
=======================================================================*/

bool eval_class (FILE *fp_det, FILE *fp_ori, const CLASSES current_class,
                const vector< vector<tGroundtruth> > &groundtruth,
                const vector< vector<tDetection> > &detections, bool compute_aos,
                double (*boxoverlap)(tDetection, tGroundtruth, int32_t, double),
                vector<double> &precision, vector<double> &aos, vector<double> &mppe,
                vector<double> &recalls_vector, const DIFFICULTY difficulty,
                const METRIC metric, FILE *fp_iour=NULL, FILE *fp_mppe=NULL,
                int analyze_recall=0, int fixed_max_z=-1, double relative_error=0.) {

  assert(groundtruth.size() == detections.size());

  // init
  int32_t n_gt=0;                                     // total no. of gt (denominator of recall)
  vector<double> v, thresholds;                       // detection scores, evaluated for recall discretization
  vector< vector<int32_t> > ignored_gt, ignored_det;  // index of ignored gt detection for current class/difficulty
  vector< vector<tGroundtruth> > dontcare;            // index of dontcare areas, included in ground truth

  std::cout << "Getting detection scores to compute thresholds" << std::endl;
  // for all test images do
  for (int32_t i=0; i<groundtruth.size(); i++){

    // holds ignored ground truth, ignored detections and dontcare areas for current frame
    vector<int32_t> i_gt, i_det;
    vector<tGroundtruth> dc;

    // only evaluate objects of current class and ignore occluded, truncated objects
    cleanData(current_class, groundtruth[i], detections[i], i_gt, dc, i_det, n_gt, difficulty, fixed_max_z);
    ignored_gt.push_back(i_gt);
    ignored_det.push_back(i_det);
    dontcare.push_back(dc);

    // compute statistics to get recall values
    tPrData pr_tmp = tPrData();
    pr_tmp = computeStatistics(current_class, groundtruth[i], detections[i], dc, i_gt, i_det, false, boxoverlap, metric, false, 0., -1., relative_error);

    // add detection scores to vector over all images
    for(int32_t j=0; j<pr_tmp.v.size(); j++)
      v.push_back(pr_tmp.v[j]);
  }

  if(n_gt <= 0){
    std::cout << "No GT samples found" << std::endl;
    return false;
  }



  // get scores that must be evaluated for recall discretization
  thresholds = getThresholds(v, n_gt);

  // compute TP,FP,FN for relevant scores
  vector<tPrData> pr;
  pr.assign(thresholds.size(),tPrData());

  // compute TP,FP,FN for relevant IOUs
  vector<tPrData> all;
  if (analyze_recall){
    all.assign(N_IOU_SAMPLE_PTS, tPrData());
  }

  std::cout << "Computing statistics" << std::endl;

  for (int32_t i=0; i<groundtruth.size(); i++){

    if (thresholds.size()-1 > 100){
      cout << "Recall discretization failed. " << thresholds.size() << " thresholds found" << endl;
      return false;
    }

    if (analyze_recall){
      // for all IOUs do:
      for(int j=0; j<N_IOU_SAMPLE_PTS; j++){
        //double iou = 0.5+(0.5/(float)(N_IOU_SAMPLE_PTS-1))*j;
        double iou = (1.0/(float)(N_IOU_SAMPLE_PTS-1))*j;
        tPrData tmp = tPrData();
        tmp  = computeStatistics(current_class, groundtruth[i], detections[i], dontcare[i],
                                ignored_gt[i], ignored_det[i], true, boxoverlap, metric,
                                compute_aos, thresholds[thresholds.size()-1], iou, relative_error);

        all[j].tp += tmp.tp;
        all[j].fn += tmp.fn;
      }
    }

    // for all scores/recall thresholds do:
    for(int32_t t=0; t<thresholds.size(); t++){
      tPrData tmp = tPrData();
      tmp = computeStatistics(current_class, groundtruth[i], detections[i], dontcare[i],
                              ignored_gt[i], ignored_det[i], true, boxoverlap, metric,
                              compute_aos, thresholds[t], -1., relative_error);

      // add no. of TP, FP, FN, AOS for current frame to total evaluation for current threshold
      pr[t].tp += tmp.tp;
      pr[t].fp += tmp.fp;
      pr[t].fn += tmp.fn;
      if(tmp.similarity!=-1){
        pr[t].similarity += tmp.similarity;
        for (int vp=0; vp<VIEWP_BINS; vp++){
          if (tmp.tp_bins[vp] != -1 && tmp.pred_bins[vp] != -1){
            pr[t].tp_bins[vp] += tmp.tp_bins[vp];
            pr[t].pred_bins[vp] += tmp.pred_bins[vp];
          }
        }
      }
    }
  }

  if (analyze_recall){
    for(float j=0; j<N_IOU_SAMPLE_PTS; j++){
      recalls_vector.push_back(all[j].tp / (double)(all[j].tp + all[j].fn));
    }
  }

  // compute recall, precision and AOS
  vector<double> recall;
  precision.assign(N_SAMPLE_PTS, 0);
  if(compute_aos){
    aos.assign(N_SAMPLE_PTS, 0);
    mppe.assign(N_SAMPLE_PTS, 0);
  }

  // compute MPPE
  double r=0;
  for (int32_t i=0; i<thresholds.size(); i++){
    r = pr[i].tp/(double)(pr[i].tp + pr[i].fn);
    recall.push_back(r);
    precision[i] = pr[i].tp/(double)(pr[i].tp + pr[i].fp);
    if(compute_aos){
      aos[i] = pr[i].similarity/(double)(pr[i].tp + pr[i].fp);
      int non_zero_bins = 0;
      for (int vp=0; vp<VIEWP_BINS; vp++){
        if (pr[i].pred_bins[vp] > 0){
          non_zero_bins++;
          mppe[i] += (pr[i].tp_bins[vp]/(double)pr[i].pred_bins[vp]);
        }
      }
      if (non_zero_bins){
        mppe[i] /= (double)non_zero_bins;
      }else{
        mppe[i] = 0;
      }
    }
  }

  // filter precision, AOS and MPPE using max_{i..end}(precision)
  for (int32_t i=0; i<thresholds.size(); i++){
    precision[i] = *max_element(precision.begin()+i, precision.end());
    if(compute_aos){
      aos[i] = *max_element(aos.begin()+i, aos.end());
      mppe[i] = *max_element(mppe.begin()+i, mppe.end());
    }
  }

  // save statisics and finish with success
  if (fixed_max_z<0) saveStats(precision, aos, recalls_vector, mppe, fp_det, fp_ori, fp_iour, fp_mppe);
  cout << "Stats computed " << endl;
  return true;
}

void saveAndPlotPlotsDist(string dir_name,string file_name,string obj_type,vector<double> vals[]){

  char command[1024];

  // save plot data to file
  FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(),"w");
  printf("Saving %s\n", (dir_name + "/" + file_name + ".txt").c_str());
  for(int dist=0; dist<vals[0].size(); dist++){

    fprintf(fp,"%f %f %f %f\n",(float)dist*DELTA_DIST+MIN_DIST,vals[0][dist],vals[1][dist],vals[2][dist]);
  }
  fclose(fp);

  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
    } else {
      fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
    }

    // set labels and ranges
    fprintf(fp,"set size ratio 0.7\n");
    fprintf(fp,"set xrange [%d:%d]\n", MIN_DIST, MAX_DIST);
    fprintf(fp,"set yrange [0:1]\n");
    fprintf(fp,"set xlabel \"Max distance\"\n");
    fprintf(fp,"set ylabel \"Recall\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

    // line width
    int32_t   lw = 5;
    if (j==0) lw = 3;

    // plot error curve
    fprintf(fp,"plot ");
    fprintf(fp,"\"%s.txt\" using 1:2 title 'Easy' with lines ls 1 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:3 title 'Moderate' with lines ls 2 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:4 title 'Hard' with lines ls 3 lw %d",file_name.c_str(),lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
    system(command);
  }
  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %s.eps %s_large.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; pdfcrop %s_large.pdf %s.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; rm %s_large.pdf",dir_name.c_str(),file_name.c_str());
  system(command);
}

void saveAndPlotPlots(string dir_name,string file_name,string obj_type,vector<double> vals[],bool is_aos,bool is_mppe=false){

  char command[1024];

  // save plot data to file
  FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(),"w");
  printf("Saving %s\n", (dir_name + "/" + file_name + ".txt").c_str());
  for (int32_t i=0; i<(int)N_SAMPLE_PTS; i++)
    fprintf(fp,"%f %f %f %f\n",(double)i/(N_SAMPLE_PTS-1.0),vals[0][i],vals[1][i],vals[2][i]);

  if (!is_mppe){
    double sum[3] = {0, 0, 0};
    double average[3] = {0, 0, 0};
    for (int v = 0; v < 3; ++v){
      for (int i=1; i<=40; i++){
        sum[v] += vals[v][i];
      }
      average[v] = sum[v]/40.0;
    }
    //fprintf(fp, "%s AP: %f %f %f\n", file_name.c_str(), average[0] * 100, average[1] * 100, average[2] * 100);
    cout << "-----------" << endl;
    printf("%s %s (%%): %.2f / %.2f / %.2f\n", file_name.c_str(), is_aos ? "AOS" : "AP", average[0] * 100, average[1] * 100, average[2] * 100);
    cout << "-----------" << endl;
  }

  fclose(fp);

  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
    } else {
      fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
    }

    // set labels and ranges
    fprintf(fp,"set size ratio 0.7\n");
    fprintf(fp,"set xrange [0:1]\n");
    fprintf(fp,"set yrange [0:1]\n");
    fprintf(fp,"set xlabel \"Recall\"\n");
    if (is_mppe) fprintf(fp,"set ylabel \"MPPE\"\n");
    else if (!is_aos) fprintf(fp,"set ylabel \"Precision\"\n");
    else         fprintf(fp,"set ylabel \"Orientation Similarity\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

    // line width
    int32_t   lw = 5;
    if (j==0) lw = 3;

    // plot error curve
    fprintf(fp,"plot ");
    fprintf(fp,"\"%s.txt\" using 1:2 title 'Easy' with lines ls 1 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:3 title 'Moderate' with lines ls 2 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:4 title 'Hard' with lines ls 3 lw %d",file_name.c_str(),lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
    system(command);
  }

  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %s.eps %s_large.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; pdfcrop %s_large.pdf %s.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; rm %s_large.pdf",dir_name.c_str(),file_name.c_str());
  system(command);
}

void saveAndPlotIOUR(string dir_name,string file_name,string obj_type,vector<double> vals[]){

  char command[1024];

  // save plot data to file
  FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(),"w");
  printf("save %s\n", (dir_name + "/" + file_name + ".txt").c_str());
  for (int32_t i=0; i<N_IOU_SAMPLE_PTS; i++){
    // For x-axis range [0.5, 1]:
    //double iou = 0.5+(0.5/(float)(N_IOU_SAMPLE_PTS-1))*i;
    // For x-axis range [0, 1]:
    double iou = (1.0/(float)(N_IOU_SAMPLE_PTS-1))*i;
    fprintf(fp,"%f %f %f %f\n",iou,vals[0][i],vals[1][i],vals[2][i]);
  }
  fclose(fp);

  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
    } else {
      fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
    }

    // set labels and ranges
    fprintf(fp,"set size ratio 0.7\n");
    // For x-axis range [0.5, 1]:
    //fprintf(fp,"set xrange [0.5:1]\n");
    // For x-axis range [0, 1]:
    fprintf(fp,"set xrange [0.1:1]\n");
    fprintf(fp,"set yrange [0:1]\n");
    fprintf(fp,"set xlabel \"IoU\"\n");
    fprintf(fp,"set ylabel \"Recall\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

    // line width
    int32_t   lw = 5;
    if (j==0) lw = 3;

    // plot error curve
    fprintf(fp,"plot ");
    fprintf(fp,"\"%s.txt\" using 1:2 title 'Easy' with lines ls 1 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:3 title 'Moderate' with lines ls 2 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:4 title 'Hard' with lines ls 3 lw %d",file_name.c_str(),lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
    system(command);
  }

  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %s.eps %s_large.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; pdfcrop %s_large.pdf %s.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; rm %s_large.pdf",dir_name.c_str(),file_name.c_str());
  system(command);
}

bool eval(string result_sha,string input_dataset,int analyze_recall,int analyze_distance){

  // set some global parameters
  initGlobals();

  // ground truth and result directories
  string gt_dir         = "data/object/label_2";
  string result_dir     = "results/" + result_sha;
  string plot_dir       = result_dir + "/plot";
  string valid_imgs_path = "lists/" + input_dataset + ".txt";

  std::cout << "Results list: " << valid_imgs_path << std::endl;

  // create output directories
  system(("mkdir " + plot_dir).c_str());

  // hold detections and ground truth in memory
  vector< vector<tGroundtruth> > groundtruth;
  vector< vector<tDetection> >   detections;

  // holds wether orientation similarity shall be computed (might be set to false while loading detections)
  // and which labels where provided by this submission
  bool compute_aos=true;
  vector<bool> eval_image(NUM_CLASS, false);
  vector<bool> eval_ground(NUM_CLASS, false);
  vector<bool> eval_3d(NUM_CLASS, false);

  std::cout << "Getting valid images... " << std::endl;
  // Get image indices
  ifstream valid_imgs(valid_imgs_path.c_str());
  if (!valid_imgs.is_open()){
    std::cout << valid_imgs_path << " not found" << std::endl;
    exit(-1);
  }
  string line;
  vector<int> indices;
  while (!valid_imgs.eof())
  {
    getline (valid_imgs,line);
    if (atoi(line.c_str())!=0){
      indices.push_back(atoi(line.c_str()));
    }
  }
  std::cout << "File loaded" << std::endl;

  N_TESTIMAGES = indices.size();

  // Just to get stats for each class
  vector<int> count, count_gt;
  count.assign(CLASS_NAMES.size(), 0);
  count_gt.assign(CLASS_NAMES.size(), 0);

  // for all images read groundtruth and detections
  std::cout << "Loading detections... " << std::endl;

  for (int32_t i=0; i<N_TESTIMAGES; i++) {

    // file name
    char file_name[256];
    switch (append_zeros){
      case 6:
        sprintf(file_name,"%06d.txt",indices[i]);
        break;
      case 8:
        sprintf(file_name,"%08d.txt",indices[i]);
        break;
      default:
        std::cout << "ERROR: Undefined number of zeros to append" << std::endl;
    }

    // read ground truth and result poses
    bool gt_success,det_success;
    vector<tGroundtruth> gt   = loadGroundtruth(gt_dir + "/" + file_name, gt_success, count_gt);
    vector<tDetection>   det  = loadDetections(result_dir + "/data/" + file_name,
                                              compute_aos, eval_image, eval_ground,
                                              eval_3d, det_success, count);
    groundtruth.push_back(gt);
    detections.push_back(det);

    // check for errors
    if (!gt_success) {
      std::cout << "ERROR: Couldn't read: " << gt_dir + "/" + file_name << " of ground truth." << std::endl;
      return false;
    }
    if (!det_success) {
      std::cout << "ERROR: Couldn't read: " << result_dir + "/data/" + file_name <<std::endl;
      return false;
    }
  }
  // Print stats
  cout << "-----------" << endl;
  cout << "GT STATS" << endl;
  cout << "-----------" << endl;
  for (int cls_idx=0; cls_idx<CLASS_NAMES.size(); cls_idx++){
    cout << CLASS_NAMES[cls_idx].c_str() << " : " << count_gt[cls_idx] << endl;
  }
  cout << "-----------" << endl;
  cout << "DET STATS" << endl;
  cout << "-----------" << endl;
  for (int cls_idx=0; cls_idx<CLASS_NAMES.size(); cls_idx++){
    cout << CLASS_NAMES[cls_idx].c_str() << " : " << count[cls_idx] << endl;
  }
  std::cout << "  done." << std::endl;

  // holds pointers for result files
  FILE *fp_det=0, *fp_ori=0, *fp_iour=0, *fp_mppe=0;

  // eval image 2D bounding boxes
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    if (count_gt[c]<=0){
      std::cout << "No ground-truth samples of " << CLASS_NAMES[c] << " found" << std::endl;
      continue;
    }
    if (eval_image[c]) {
      cout << "Starting 2D evaluation (" << CLASS_NAMES[c] << ") ..." << endl;
      fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection.txt").c_str(), "w");
      fp_iour = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_iour.txt").c_str(), "w");
      if(compute_aos) {
        fp_ori = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_orientation.txt").c_str(),"w");
        fp_mppe = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_mppe.txt").c_str(), "w");
      }
      vector<double> precision[4], aos[4], mppe[4], recalls[4];
      if(   !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precision[0], aos[0], mppe[0], recalls[0], EASY, IMAGE, fp_iour, fp_mppe, analyze_recall)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precision[1], aos[1], mppe[1], recalls[1], MODERATE, IMAGE, fp_iour, fp_mppe, analyze_recall)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precision[2], aos[2], mppe[2], recalls[2], HARD, IMAGE, fp_iour, fp_mppe, analyze_recall)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precision[3], aos[3], mppe[3], recalls[3], ALL, IMAGE, fp_iour, fp_mppe, analyze_recall)) {
        cout << CLASS_NAMES[c].c_str() << " evaluation failed." << endl;
        return false;
      }
      fclose(fp_det);
      fclose(fp_iour);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection", CLASS_NAMES[c], precision, 0);
      if (analyze_recall){
        saveAndPlotIOUR(plot_dir, CLASS_NAMES[c] + "_iour", CLASS_NAMES[c], recalls);
      }
      if(compute_aos){
        saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_orientation", CLASS_NAMES[c], aos, 1);
        saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_mppe", CLASS_NAMES[c], mppe, 1, true);
        fclose(fp_mppe);
        fclose(fp_ori);
      }

      // Recall vs distance
      if (analyze_distance){
        vector<double> recall_per_distance[4];
        for(int dist=MIN_DIST; dist<=MAX_DIST; dist+=DELTA_DIST){
          vector<double> precisionD[4], aosD[4], mppeD[4], recallsD[4];
          if(   !eval_class(NULL, NULL, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precisionD[0], aosD[0], mppeD[0], recallsD[0], EASY, IMAGE, NULL, NULL, analyze_recall, dist)
             || !eval_class(NULL, NULL, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precisionD[1], aosD[1], mppeD[1], recallsD[1], MODERATE, IMAGE, NULL, NULL, analyze_recall, dist)
             || !eval_class(NULL, NULL, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precisionD[2], aosD[2], mppeD[2], recallsD[2], HARD, IMAGE, NULL, NULL, analyze_recall, dist)
             || !eval_class(NULL, NULL, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precisionD[3], aosD[3], mppeD[3], recallsD[3], ALL, IMAGE, NULL, NULL, analyze_recall, dist)) {
            cout << CLASS_NAMES[c].c_str() << " evaluation failed." << endl;
            return false;
          }
          for (int idx=0; idx<4; idx++){
            // TODO: Improve column indexing
            // idx=0 (Car) -> required IoU=0.7: 0.5+20*((1-0.5)/(N_IOU_SAMPLE_PTS-1))
            // idx>0 (Pedestrian, Cyclist) -> required IoU=0.5: 0.5+0
            recall_per_distance[idx].push_back(recallsD[idx][idx==0?20:0]); //TODO TODO TODO
          }
        }
        saveAndPlotPlotsDist(plot_dir, CLASS_NAMES[c] + "_dist", CLASS_NAMES[c], recall_per_distance);
        cout << "  done." << endl;
      }

    }else{
      std::cout << "Found no " << CLASS_NAMES[c] << " detections" << std::endl;
    }
  }

  // eval image 2D bounding boxes with relative error
  for (int re = 0; re < NUM_RELATIVE_ERROR; re++) {
    for (int c = 0; c < NUM_CLASS; c++) {
      CLASSES cls = (CLASSES)c;
      if (count_gt[c]<=0){
        std::cout << "No ground-truth samples of " << CLASS_NAMES[c] << " found" << std::endl;
        continue;
      }
      double relative_error = MAX_RELATIVE_ERROR[re][c];
      if (eval_image[c]) {
        stringstream ss;
        ss << relative_error*100;
        string err_percent = ss.str();
        cout << "Starting 2D evaluation with " + err_percent + "% relative error (" << CLASS_NAMES[c] << ") ..." << endl;
        fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection_" + err_percent + "%.txt").c_str(), "w");
        fp_iour = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_iour_" + err_percent + "%.txt").c_str(), "w");
        if(compute_aos) {
          fp_ori = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_orientation_" + err_percent + "%.txt").c_str(),"w");
          fp_mppe = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_mppe_" + err_percent + "%.txt").c_str(), "w");
        }
        vector<double> precision[4], aos[4], mppe[4], recalls[4];
        if(   !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlapWithRelativeError, precision[0], aos[0], mppe[0], recalls[0], EASY, IMAGE, fp_iour, fp_mppe, analyze_recall, -1, relative_error)
           || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlapWithRelativeError, precision[1], aos[1], mppe[1], recalls[1], MODERATE, IMAGE, fp_iour, fp_mppe, analyze_recall, -1, relative_error)
           || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlapWithRelativeError, precision[2], aos[2], mppe[2], recalls[2], HARD, IMAGE, fp_iour, fp_mppe, analyze_recall, -1, relative_error)
           || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlapWithRelativeError, precision[3], aos[3], mppe[3], recalls[3], ALL, IMAGE, fp_iour, fp_mppe, analyze_recall, -1, relative_error)) {
          cout << CLASS_NAMES[c].c_str() << " evaluation failed." << endl;
          return false;
        }
        fclose(fp_det);
        fclose(fp_iour);
        saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection_" + err_percent + "%", CLASS_NAMES[c], precision, 0);
        if (analyze_recall){
          saveAndPlotIOUR(plot_dir, CLASS_NAMES[c] + "_iour_" + err_percent + "%", CLASS_NAMES[c], recalls);
        }
        if(compute_aos){
          saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_orientation_" + err_percent + "%", CLASS_NAMES[c], aos, 1);
          saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_mppe_" + err_percent + "%", CLASS_NAMES[c], mppe, 1, true);
          fclose(fp_mppe);
          fclose(fp_ori);
        }

        // Recall vs distance
        if (analyze_distance){
          vector<double> recall_per_distance[4];
          for(int dist=MIN_DIST; dist<=MAX_DIST; dist+=DELTA_DIST){
            vector<double> precisionD[4], aosD[4], mppeD[4], recallsD[4];
            if(   !eval_class(NULL, NULL, cls, groundtruth, detections, compute_aos, imageBoxOverlapWithRelativeError, precisionD[0], aosD[0], mppeD[0], recallsD[0], EASY, IMAGE, NULL, NULL, analyze_recall, dist, relative_error)
               || !eval_class(NULL, NULL, cls, groundtruth, detections, compute_aos, imageBoxOverlapWithRelativeError, precisionD[1], aosD[1], mppeD[1], recallsD[1], MODERATE, IMAGE, NULL, NULL, analyze_recall, dist, relative_error)
               || !eval_class(NULL, NULL, cls, groundtruth, detections, compute_aos, imageBoxOverlapWithRelativeError, precisionD[2], aosD[2], mppeD[2], recallsD[2], HARD, IMAGE, NULL, NULL, analyze_recall, dist, relative_error)
               || !eval_class(NULL, NULL, cls, groundtruth, detections, compute_aos, imageBoxOverlapWithRelativeError, precisionD[3], aosD[3], mppeD[3], recallsD[3], ALL, IMAGE, NULL, NULL, analyze_recall, dist, relative_error)) {
              cout << CLASS_NAMES[c].c_str() << " evaluation failed." << endl;
              return false;
            }
            for (int idx=0; idx<4; idx++){
              // TODO: Improve column indexing
              // idx=0 (Car) -> required IoU=0.7: 0.5+20*((1-0.5)/(N_IOU_SAMPLE_PTS-1))
              // idx>0 (Pedestrian, Cyclist) -> required IoU=0.5: 0.5+0
              recall_per_distance[idx].push_back(recallsD[idx][idx==0?20:0]); //TODO TODO TODO
            }
          }
          saveAndPlotPlotsDist(plot_dir, CLASS_NAMES[c] + "_dist_" + err_percent + "%", CLASS_NAMES[c], recall_per_distance);
          cout << "  done." << endl;
        }

      }else{
        std::cout << "Found no " << CLASS_NAMES[c] << " detections" << std::endl;
      }
    }
  }

  // don't evaluate AOS for birdview boxes and 3D boxes
  compute_aos = false;

  // eval bird's eye view bounding boxes
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    if (count_gt[c]<=0){
      std::cout << "Found no " << CLASS_NAMES[c] << " ground-truth samples" << std::endl;
      continue;
    }
    if (eval_ground[c]) {
      cout << "Starting bird's eye evaluation (" << CLASS_NAMES[c] << ") ...";
      fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection_ground.txt").c_str(), "w");
      fp_iour = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_iour_ground.txt").c_str(), "w");
      vector<double> precision[4], aos[4], mppe[4], recalls[4];
      if(   !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, groundBoxOverlap, precision[0], aos[0], mppe[0], recalls[0], EASY, GROUND, fp_iour, NULL, analyze_recall)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, groundBoxOverlap, precision[1], aos[1], mppe[1], recalls[1], MODERATE, GROUND, fp_iour, NULL, analyze_recall)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, groundBoxOverlap, precision[2], aos[2], mppe[2], recalls[2], HARD, GROUND, fp_iour, NULL, analyze_recall)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, groundBoxOverlap, precision[3], aos[3], mppe[3], recalls[3], ALL, GROUND, fp_iour, NULL, analyze_recall)) {
        cout << CLASS_NAMES[c].c_str() << " evaluation failed." << endl;
        return false;
      }
      fclose(fp_det);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection_ground", CLASS_NAMES[c], precision, 0);
      if (analyze_recall){
        saveAndPlotIOUR(plot_dir, CLASS_NAMES[c] + "_iour_ground", CLASS_NAMES[c], recalls);
      }
      cout << "  done." << endl;
    }
  }

  // eval 3D bounding boxes
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    if (count_gt[c]<=0){
      std::cout << "Found no " << CLASS_NAMES[c] << " ground-truth samples" << std::endl;
      continue;
    }
    if (eval_3d[c]) {
      cout << "Starting 3D evaluation (" << CLASS_NAMES[c] << ") ..." << endl;
      fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection_3d.txt").c_str(), "w");
      fp_iour = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_iour_3d.txt").c_str(), "w");
      vector<double> precision[4], aos[4], mppe[4], recalls[4];
      if(   !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, box3DOverlap, precision[0], aos[0], mppe[0], recalls[0], EASY, BOX3D, fp_iour, NULL, analyze_recall)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, box3DOverlap, precision[1], aos[1], mppe[1], recalls[1], MODERATE, BOX3D, fp_iour, NULL, analyze_recall)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, box3DOverlap, precision[2], aos[2], mppe[2], recalls[2], HARD, BOX3D, fp_iour, NULL, analyze_recall)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, box3DOverlap, precision[3], aos[3], mppe[3], recalls[3], ALL, BOX3D, fp_iour, NULL, analyze_recall)) {
        cout << CLASS_NAMES[c].c_str() << " evaluation failed." << endl;
        return false;
      }
      fclose(fp_det);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection_3d", CLASS_NAMES[c], precision, 0);
      if (analyze_recall){
        saveAndPlotIOUR(plot_dir, CLASS_NAMES[c] + "_iour_3d", CLASS_NAMES[c], recalls);
      }
      cout << "  done." << endl;
    }
  }

  // success
  return true;
}

int32_t main (int32_t argc,char *argv[]) {

  if (argc<3 || argc>4) {
    cout << "Usage: ./eval_detection result_sha val_dataset [analyze_recall (default=0)] [analyze_distance (default=0)]" << endl;
    return 1;
  }

  // read arguments
  string result_sha = argv[1];
  string input_dataset = argv[2];

  // Obtain Recall vs IOU graph
  int analyze_recall=0;
  if (argc==4 || argc==5){
    string third_parameter = argv[3];
    analyze_recall = atoi(third_parameter.c_str());
  }

  // Obtain Recall vs distance graph
  int analyze_distance=0;
  if (argc==5){
    string fourth_parameter = argv[4];
    analyze_distance = atoi(fourth_parameter.c_str());
  }

  std:cout << "Starting evaluation..." << std::endl;

  // run evaluation
  if(eval(result_sha,input_dataset,analyze_recall,analyze_distance)){
    cout << "Evaluation finished successfully" << endl;
  }else{
    cout << "Something happened..." << endl;
  };

  return 0;
}