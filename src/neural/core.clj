
(ns neural.core
  (:gen-class))

(use 'clojure.core.matrix)

(require '[clojure.data.csv :as csv] '[clojure.java.io :as io])

(defn label-to-vec "turns a label into a vector of zeros with the label index 1, e.g. 5 --> [0 0 0 0 0 1 0 0 0 0]" [label]
  (loop [index 0 vec [] ]
    (if (> index 9) vec
   (recur (inc index)  (conj vec (if (== index label) 1 0))))
  ))

(defn read-row [reader row-index]
  (let [data (csv/read-csv reader)]
    (nth data row-index)))

(defn make-batch [filename start-row number]
  
(with-open [reader (io/reader filename)]
(let [data (csv/read-csv reader)]
  (loop [index start-row result []]
    (if (>= index (+ start-row number)) result 
        (recur (inc index) (conj result (vec (let [row-data (->>  (nth data index)
                                                                  (map #(Long/parseLong %)))]
                                               [(rest row-data) (label-to-vec (first row-data))])))))
)
    )))

 

(defn mmul2 "makes an n x m matrix out of n element vec1 and m element vec2" [vec1 vec2]
  (mmul (transpose [vec1]) [vec2]) )



(defn activate 
"Sigmoid activation function"
[a]
  (into [] (map (fn [x] (/ 1. (+ 1 (Math/exp (- 0 x))))) a)))

(defn sigmoid-prime [a] (let [b (activate a)] (mul b (sub 1 b))))

(def layer-sizes [784,30,10])

(def weight-shapes (into [] (zipmap (rest layer-sizes) (drop-last layer-sizes))))


(defn randmat "Creates a random matrix of shape sh" [sh]  (reshape (repeatedly (* (sh 0) (sh 1)) #(/ (rand)  (sh 1))) sh))


(def weights  (into [] (for [ashape weight-shapes] (randmat ashape)))  )

(def biases (into [] (for [size (rest layer-sizes)] (vec (repeat size 0))))
)

(defn propagate1
"use weights and bias for one layer"
[x [w b]]
  ( activate (add (mmul w x) b)))

(defn propagate "propagate weights and biases from the input layer to the output layer"
  [x [w b]]
(reduce propagate1 x (into [] (zipmap w b)))
)

(defn accuracy [predictions label] (== label (.indexOf predictions (apply max predictions)))
)

(defn matsq "square matrix multiplication "[x] (mmul x x)) 

(defn cost [a y] (/ (matsq (sub a y)) (* 2 (count a))))

(defn dcost "cost derivative with respect to output" 
[a y] (sub a y)
)

(defn delta [z y] (let [a (activate z)] (mul (dcost a y) (mul (sub 1 a) a)) ))

(defn propadd
"use weights and bias for one layer and add to the list of layers"
[layers [w b]]
  (into  layers [ (add (mmul w (last layers)) b)])
)

(defn deltal [w z delta]
  (mul (mmul (transpose w) delta) (sigmoid-prime z))
)

(defn backprop [x w b y]
  (let [layers  (reduce propadd [x] (into [] (zipmap w b))) 
        ]
    (loop [layer (- (count layer-sizes) 2) result (conj '() (delta (last layers) y))]

      (if (< layer 1)
        [(into [] result) layers]

        (recur (dec layer)  (conj result 
                                        (deltal (weights layer) (layers layer) (first result)) 
                      
 ))))
))

(defn grad-desc "gradient descent for one sample" [[b w] sample]
 (let  [x (first sample) y (second sample) bp (backprop x w b y) eta 1] ;x input y labels

   [(map sub b (mul eta (first bp) )      ) (map sub w (let [[bs layers] bp] (mul eta (map mmul2 bs (drop-last layers) ))))]))



(defn update-mini-batch "update weights and biases applying gradient descent using backprop"  [mini-batch eta]
 
  (reduce grad-desc [biases weights] mini-batch))

  (defn -main
    "main"
    [& args]
    (propagate args))
