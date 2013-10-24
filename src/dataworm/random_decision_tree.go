package dataworm

import (
	"math/rand"
	"sync"
	"fmt"
)

type TreeNode struct {
	left, right, depth int
	prediction         float64
	samples            []int
	feature_split	Feature
}

func (t *TreeNode) AddSample(k int) {
	t.samples = append(t.samples, k)
}

type Tree struct {
	nodes []*TreeNode
}

func (t *Tree) AddTreeNode(n *TreeNode) {
	t.nodes = append(t.nodes, n)
}

func (t *Tree) Size() int{
	return len(t.nodes)
}

func (t *Tree) GetNode(i int) *TreeNode {
	return t.nodes[i]
}

func GoLeft(sample MapBasedSample, feature_split Feature) bool {
	value, ok := sample.Features[feature_split.Id]
	if ok && value >= feature_split.Value {
		return true
	} else {
		return false
	}
}

func SingleRandomDTBuild(samples []MapBasedSample, feature_splits []Feature) Tree {
	tree := Tree{}
	queue := make(chan *TreeNode, 1024)
	root := TreeNode{depth: 0, left: -1, right: -1, prediction: -1, samples: []int{}}
	for i, _ := range samples {
		root.AddSample(i)
	}

	queue <- &root
	tree.AddTreeNode(&root)
	for {
		if len(queue) == 0 {
			break
		}

		node := <-queue
		positive := 0.0
		total := 0.0
		for _, k := range node.samples {
			positive += samples[k].LabelDoubleValue()
			total += 1.0
		}
		node.prediction = positive / total
		
		if node.depth >= len(feature_splits){
			continue
		}
		node.feature_split = feature_splits[node.depth]
		left_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: -1, samples: []int{}}
		right_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: -1, samples: []int{}}
		
		for _, k := range node.samples {
			positive += samples[k].LabelDoubleValue()
			total += 1.0
			if GoLeft(samples[k], feature_splits[node.depth]) {
				left_node.samples = append(left_node.samples, k)
			} else {
				right_node.samples = append(right_node.samples, k)
			}
		}
		node.samples = nil

		if len(left_node.samples) > 10 {
			queue <- &left_node
			node.left = len(tree.nodes)
			tree.AddTreeNode(&left_node)
		}

		if len(right_node.samples) > 10 {
			queue <- &right_node
			node.right = len(tree.nodes)
			tree.AddTreeNode(&right_node)
		}
	}
	return tree
}

func RandomTreePrediction(tree Tree, sample MapBasedSample) float64 {
	node := tree.GetNode(0)
	for {
		if GoLeft(sample, node.feature_split){
			if node.left >= 0 && node.left < tree.Size(){
				node = tree.GetNode(node.left)
			} else {
				return node.prediction
			}
		} else {
			if node.right >= 0 && node.right < tree.Size(){
				node = tree.GetNode(node.right)
			} else {
				return node.prediction
			}
		}
	}
	return node.prediction
}

func RandomDecisionTreeBuild(samples []MapBasedSample) []Tree {

	forest := []Tree{}
	for k := 0; k < 30; k++ {
		m := rand.Int() % len(samples)
		random_sample := samples[m]
		feature_split := []Feature{}
		for fid, fvalue := range random_sample.Features {
			feature_split = append(feature_split, Feature{Id: fid, Value: fvalue})
		}
		tree := SingleRandomDTBuild(samples, feature_split)
		forest = append(forest, tree)
	}
	return forest
}

func RandomDecisionTree(train_path string, test_path string) {
	train_dataset := DataSet{}
	train_dataset.Samples = make(chan Sample, 1000)
	samples := []MapBasedSample{}
	var wait sync.WaitGroup
	wait.Add(2)
	var err error
	go func() {
		err = train_dataset.Load(train_path, -1, 1)
		wait.Done()
	}()

	go func() {
		for sample := range train_dataset.Samples {
			samples = append(samples, sample.ToMapBasedSample())
		}
		wait.Done()
	}()

	wait.Wait()

	forest := RandomDecisionTreeBuild(samples)
	
	test_dataset := DataSet{}
	test_dataset.Samples = make(chan Sample, 1000)
	
	wait.Add(2)
	go func() {
		err = test_dataset.Load(test_path, -1, 1)
		wait.Done()
	}()
	
	predictions := []LabelPrediction{}
	go func() {
		for sample := range test_dataset.Samples {
			msample := sample.ToMapBasedSample()
			prediction := 0.0
			total := 0.0
			for _, tree := range forest{
				prediction += RandomTreePrediction(tree, msample)
				total += 1.0	
			}
			prediction /= total
			predictions = append(predictions, LabelPrediction{Label: sample.Label, Prediction: prediction})
		}
		wait.Done()
	}()
	
	wait.Wait()
	fmt.Println("AUC")
	fmt.Println(AUC(predictions))
}
