package dataworm

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type TreeNode struct {
	left, right, depth int
	prediction         float64
	samples            []int
	feature_split      Feature
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

func (t *Tree) Size() int {
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

func GetElementFromQueue(queue chan *TreeNode, n int) []*TreeNode {
	ret := []*TreeNode{}
	for i := 0; i < n; i++ {
		if len(queue) == 0 {
			time.Sleep(1e9)
			if len(queue) == 0 {
				break
			}
		}
		node := <-queue
		ret = append(ret, node)
	}
	return ret
}

func AppendNodeToTree(samples []MapBasedSample, node *TreeNode, queue chan *TreeNode, tree *Tree, p *RDTParams, feature_splits []Feature) {
	positive := 0.0
	total := 0.0
	for _, k := range node.samples {
		positive += samples[k].LabelDoubleValue()
		total += 1.0
	}
	node.prediction = positive / total

	if node.depth >= len(feature_splits) {
		return
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

	if len(left_node.samples) > p.MinLeafSize {
		queue <- &left_node
		node.left = len(tree.nodes)
		tree.AddTreeNode(&left_node)
	}

	if len(right_node.samples) > p.MinLeafSize {
		queue <- &right_node
		node.right = len(tree.nodes)
		tree.AddTreeNode(&right_node)
	}
}

func SingleRandomDTBuild(samples []MapBasedSample, feature_splits []Feature, p RDTParams) Tree {
	tree := Tree{}
	queue := make(chan *TreeNode, 1024)
	root := TreeNode{depth: 0, left: -1, right: -1, prediction: -1, samples: []int{}}
	for i, _ := range samples {
		root.AddSample(i)
	}

	queue <- &root
	tree.AddTreeNode(&root)
	for {
		nodes := GetElementFromQueue(queue, 10)
		if len(nodes) == 0 {
			break
		}

		for _, node := range nodes {
			AppendNodeToTree(samples, node, queue, &tree, &p, feature_splits)
		}
	}
	return tree
}

func RandomTreePrediction(tree *Tree, sample MapBasedSample) float64 {
	node := tree.GetNode(0)
	for {
		if GoLeft(sample, node.feature_split) {
			if node.left >= 0 && node.left < tree.Size() {
				node = tree.GetNode(node.left)
			} else {
				return node.prediction
			}
		} else {
			if node.right >= 0 && node.right < tree.Size() {
				node = tree.GetNode(node.right)
			} else {
				return node.prediction
			}
		}
	}
	return node.prediction
}

type RDTParams struct {
	TreeCount   int
	MinLeafSize int
}

func RandomDecisionTreeBuild(samples []MapBasedSample, p RDTParams) chan *Tree {
	forest := make(chan *Tree, p.TreeCount)
	var wait sync.WaitGroup
	wait.Add(p.TreeCount)
	for k := 0; k < p.TreeCount; k++ {
		go func() {
			m := rand.Int() % len(samples)
			random_sample := samples[m]
			feature_split := []Feature{}
			for fid, fvalue := range random_sample.Features {
				feature_split = append(feature_split, Feature{Id: fid, Value: fvalue})
			}
			tree := SingleRandomDTBuild(samples, feature_split, p)
			forest <- &tree
			wait.Done()
		}()
	}
	wait.Wait()
	close(forest)
	return forest
}

func RandomDecisionTree(train_path string, test_path string, p RDTParams) {
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

	forest_chan := RandomDecisionTreeBuild(samples, p)

	forest := []*Tree{}

	for tree := range forest_chan {
		forest = append(forest, tree)
	}

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
			for _, tree := range forest {
				prediction += RandomTreePrediction(tree, msample)
				total += 1.0
			}
			prediction /= total
			//fmt.Println(prediction)
			predictions = append(predictions, LabelPrediction{Label: sample.Label, Prediction: prediction})
		}
		wait.Done()
	}()

	wait.Wait()
	fmt.Println("AUC")
	fmt.Println(AUC(predictions))
}
