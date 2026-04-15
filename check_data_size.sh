
fn(){
  for f in  $1/* ; do echo $f; ls -l $f | wc -l ; done
}
fn "./data/processed_all"
fn "./data/filtered_data"
#fn "./data/need_review_semantic_ground"
#fn "./data/need_review_semantic_sky"
